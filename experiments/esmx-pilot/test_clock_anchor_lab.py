"""Check that the desktop comparison retains order and honors the replay gate."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from clock_compact_lab import Detector, tensor
from clock_elastic_lab import FRAMES, align, cepstra, example_id, predict


class ClockAnchorLabTest(TestCase):
    def test_whole_turn_alignment_prefers_matching_order(self):
        first = np.linspace(-1, 1, FRAMES)[:, None]
        result = align(first, np.stack([first, first[::-1]]))
        self.assertEqual(result[0], 0)
        self.assertGreater(result[1], 0.5)

    def test_constant_spectral_bias_is_removed(self):
        sample = np.random.default_rng(9).integers(-30, 30, (33, 20), dtype=np.int8)
        bias = np.arange(20, dtype=np.int8)
        np.testing.assert_allclose(cepstra(sample, True), cepstra(sample + bias, True), atol=1e-12)
        np.testing.assert_allclose(tensor(sample)[0], tensor(sample + bias)[0], atol=1e-6)

    def test_veto_is_applied_after_scoring(self):
        frozen = {"arms": {"test": {"templates": {}, "delta": False, "rule": {"threshold": 0.25}}}}
        with (patch("clock_elastic_lab.matrix", return_value=np.zeros((13, 20))),
              patch("clock_elastic_lab.feature_matrix", return_value=np.zeros(9)),
              patch("clock_elastic_lab.score", return_value=1.0)):
            for decision in ("dice_negation_conflict", "fact_veto", "no_match"):
                result = predict(SimpleNamespace(frontend=None), frozen, b"", decision)["test"]
                self.assertTrue(result["proposal"])
                self.assertEqual(result["eligible"], decision == "no_match")

    def test_recording_id_rejects_parent_paths(self):
        with self.assertRaises(ValueError):
            example_id("../T02-388", r"[CT][0-9]{2}-[0-9]{3}")

    def test_compact_output_preserves_batch_axis(self):
        detector = Detector().eval()
        features = torch.zeros((2, 20, 64))
        lengths = torch.zeros(2)
        with torch.inference_mode():
            logits = detector(features, lengths)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())
