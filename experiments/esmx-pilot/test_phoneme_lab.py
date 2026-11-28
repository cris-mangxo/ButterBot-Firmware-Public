"""Check temporal geometry and rejection behavior for the phoneme prototype."""

import json
from hashlib import sha256
import unittest

import numpy as np
import torch

from phoneme_student_lab import WIDTH, Student, gate, quantize
from phoneme_teacher_lab import align_frontend, collapse, edit_distance, nearest_intent, project
from phoneme_replay_lab import overlaps_adjustment, summarize


class PhonemeLabTest(unittest.TestCase):
    def test_ctc_blank_separates_identical_phones(self):
        decoded = collapse(np.array([0, 4, 4, 0, 4, 6, 0]).tolist())
        self.assertEqual(decoded, (4, 4, 6))
        self.assertEqual(json.loads(json.dumps(edit_distance(decoded, np.array([4, 6])))), 1)

    def test_teacher_projection_keeps_blank_out_of_other(self):
        distribution = project(torch.zeros((1, 4)), torch.tensor([0, 1, 2, 2]), 3)
        np.testing.assert_allclose(distribution, [[0.25, 0.25, 0.5]])

    def test_interpolation_uses_frontend_centers(self):
        probs = np.array([[1, 0], [0, 1]], dtype=np.float32)
        np.testing.assert_allclose(align_frontend(probs, 2), [[0.875, 0.125], [0, 1]])

    def test_equal_competing_phrases_abstain(self):
        templates = [((1, 2, 4), 6), ((1, 2, 5), 1)]
        self.assertEqual(nearest_intent((1, 2, 3), templates), 0)
        self.assertEqual(nearest_intent((1, 2, 4), templates), 6)

    def test_negation_veto_can_suppress_strong_intent(self):
        intents = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]]).repeat(3, 1)
        scores = gate(intents, torch.tensor([1.0, -1.0, 0.0]))
        self.assertEqual(scores.tolist(), [0, 4, 0])

    def test_padding_does_not_change_valid_frames_or_decision(self):
        torch.manual_seed(123)
        model = Student(33).eval()
        pcm = torch.randn(1, 20, 20)
        with torch.inference_mode():
            exact = model(pcm, torch.tensor([20]))
            extended = model(torch.nn.functional.pad(pcm, (0, 13)), torch.tensor([20]))
        torch.testing.assert_close(exact[0], extended[0][:, :20])
        torch.testing.assert_close(exact[1], extended[1])
        torch.testing.assert_close(exact[2], extended[2])

    def test_weight_export_has_int8_weights_and_stays_in_budget(self):
        model = Student(33).eval()
        arrays, restored = quantize(model)
        self.assertLess(sum(item.nbytes for item in arrays.values()), 10_240)
        self.assertEqual(arrays["input.weight"].dtype, np.int8)
        self.assertEqual(arrays["input.weight.scale"].shape, (WIDTH, 1, 1))
        error = (restored.input.weight - model.input.weight).abs()
        self.assertLessEqual(float(error.detach().max()), float(arrays["input.weight.scale"].max()) / 2 + 1e-6)

    def test_replay_records_waveform_overlap_and_wrong_intent(self):
        pcm = np.full(640, 1600, dtype="<i2")
        match = sha256((pcm.astype(np.float32) / 32_768).tobytes()).hexdigest()
        self.assertTrue(overlaps_adjustment(pcm.tobytes(), {match}))
        self.assertFalse(overlaps_adjustment((pcm + 1).tobytes(), {match}))
        rows = [
            {"id": "one", "expected": 6, "raw": {
                "baseline": 0, "candidate": 6, "combined": 6, "matches_adjustment_pcm": True}},
            {"id": "two", "expected": 0, "raw": {
                "baseline": 0, "candidate": 5, "combined": 5, "matches_adjustment_pcm": False}},
        ]
        report = summarize(rows, "raw")
        self.assertEqual(report["new_correct"], ["one"])
        self.assertEqual(report["new_wrong"], ["two"])
        self.assertEqual(report["new_wrong_intent"], {5: 1})
        self.assertEqual(report["matching_adjustment_pcm"], ["one"])


if __name__ == "__main__":
    unittest.main()
