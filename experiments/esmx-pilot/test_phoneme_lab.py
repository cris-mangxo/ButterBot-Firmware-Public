"""Check temporal geometry and rejection behavior for the phoneme prototype."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compact_d6_training_lab import accepted as compact_accepted, calibrate as calibrate_compact_d6
from compact_d6_training_lab import context_accepted, select_context_limits
from d6_teacher_guard_lab import Scorer, accepted as d6_accepted, eligible_pcm, permitted, scope_cap, select_limit
from d6_teacher_guard_lab import exterior_phone_counts
from phoneme_student_lab import WIDTH, Student, gate, quantize
from phoneme_teacher_lab import align_frontend, collapse, edit_distance, nearest_intent, project
from phoneme_replay_lab import overlaps_adjustment, strip_digital_edges, summarize
from phoneme_reject_lab import load_negatives
from phoneme_confirm_lab import checked_bank, confirmed, local_ctc_alignment, local_ctc_cost
from phoneme_ctc_adapt_lab import METHOD as ADAPTATION, TARGET_SHA256
from time_lab import file_digest


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

    def test_negative_cache_excludes_reserved_speakers_even_with_valid_archive_seal(self):
        with TemporaryDirectory() as work:
            directory = Path(work)
            split = directory / "speakers.json"
            split.write_text(json.dumps({"parquet_sha256": "corpus", "speaker_roles": {"a": "heldout"}}))
            item = directory / "clip.npz"
            np.savez_compressed(item, features=np.zeros((8, 20), dtype=np.int8))
            manifest = directory / "index.json"
            manifest.write_text(json.dumps({
                "schema": "bb-chm150-negative-frames-1", "split_sha256": file_digest(split),
                "source_sha256": "corpus", "extractor_sha256": "frontend",
                "records": {"clip": {"split": "train", "speaker": "a", "frames": 8,
                                     "label": "out_of_domain", "feature_sha256": file_digest(item)}},
            }))
            with patch("phoneme_reject_lab.NEGATIVE_CACHE_SHA256", file_digest(manifest)):
                with self.assertRaisesRegex(ValueError, "designated training or development speaker"):
                    load_negatives(directory, split, "frontend")

    def test_replay_detects_pcm_overlap_in_native_human_cache(self):
        pcm = np.full(640, 1600, dtype="<i2").tobytes()
        self.assertTrue(overlaps_adjustment(pcm, set(), {sha256(pcm).hexdigest()}))

    def test_digital_edge_alignment_keeps_every_sample_inside_the_turn(self):
        body = np.array([1, 0, 0, -1, 0, 230], dtype="<i2").tobytes()
        original = bytes(640) + body + bytes(320)
        capture = bytes(960) + body + bytes(960)
        self.assertEqual(strip_digital_edges(original), body)
        self.assertEqual(strip_digital_edges(capture), body)
        self.assertEqual(strip_digital_edges(bytes(128_000)), b"")

    def test_digital_edge_alignment_cannot_recover_an_oversized_turn(self):
        overlong = b"\x01\x00" + bytes(128_000)
        self.assertEqual(strip_digital_edges(overlong), overlong)

    def test_local_ctc_ignores_unrelated_leading_and_trailing_frames(self):
        logits = np.array([[0, 0, 0, 9], [0, 9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 9]], dtype=np.float32)
        self.assertEqual(local_ctc_cost(logits, (1, 2)), 0.0)
        self.assertGreater(local_ctc_cost(logits[1:3], (2, 1)), 0)

    def test_local_ctc_repeated_symbols_need_a_blank(self):
        logits = np.array([[0, 9], [9, 0], [0, 9]], dtype=np.float32)
        self.assertEqual(local_ctc_cost(logits, (1, 1)), 0.0)
        self.assertTrue(np.isinf(local_ctc_cost(logits[[0, 2]], (1, 1))))

    def test_local_ctc_aligns_inside_a_turn_without_changing_cost(self):
        logits = np.array([[0, 0, 0, 9], [0, 9, 0, 0], [9, 0, 0, 0],
                           [0, 0, 9, 0], [0, 0, 0, 9]], dtype=np.float32)
        aligned = local_ctc_alignment(logits, (1, 2), track_span=True)
        self.assertEqual(aligned, (local_ctc_cost(logits, (1, 2)), (1, 4)))
        score, interval = local_ctc_alignment(logits[1:2], (1, 1), track_span=True)
        self.assertTrue(np.isinf(score))
        self.assertIsNone(interval)

    def test_lexical_veto_only_removes_an_intention(self):
        cutoff = {5: 2.0}
        self.assertEqual(confirmed(0, None, cutoff), 0)
        self.assertEqual(confirmed(5, 2.5, cutoff), 0)
        self.assertEqual(confirmed(5, 2.0, cutoff), 5)

    def test_d6_reference_calibration_rejects_ties_with_controls(self):
        def score(positive, cost, margin, veto=False):
            return {"positive": positive, "score": {"cost": cost, "margin": margin, "veto": veto}}

        development = [score(True, 0.25, 0.5), score(True, 1.0, 0.5),
                       score(False, 1.0, 0.5), score(False, 0.0, 1.0, veto=True)]
        limit, margin = select_limit(development)
        self.assertEqual(limit, 0.25)
        self.assertTrue(d6_accepted(development[0]["score"], limit, margin))
        self.assertFalse(d6_accepted(development[2]["score"], limit, margin))
        self.assertFalse(d6_accepted(development[3]["score"], limit, margin))

    def test_d6_reference_reports_no_fit_for_indistinguishable_utterances(self):
        score = {"cost": 0.5, "margin": 1.0, "veto": False}
        development = [{"positive": True, "score": score}, {"positive": False, "score": score}]
        self.assertEqual(select_limit(development), (None, None))

    def test_d6_reference_skips_inference_for_invalid_pcm(self):
        scorer = Scorer.__new__(Scorer)
        invalid = (b"", bytes(127_999), bytes(128_002))
        with patch.object(scorer, "predict") as predict:
            for sample in invalid:
                self.assertFalse(eligible_pcm(sample))
                self.assertIsNone(scorer.run(sample))
            predict.assert_not_called()
        self.assertTrue(eligible_pcm(bytes(128_000)))

    def test_d6_scope_has_no_fit_when_a_control_has_no_exterior_phones(self):
        valid = {"scope": {"before": 1, "after": 2}, "cost": 0.3, "margin": 0.5, "veto": False}
        control = dict(valid, scope={"before": 0, "after": 0})
        orders = [{"positive": True, "score": valid}]
        self.assertIsNone(scope_cap(orders + [{"positive": False, "score": control}]))
        control["scope"]["after"] = 7
        cap = scope_cap(orders + [{"positive": False, "score": control}])
        self.assertEqual(cap, 4)
        self.assertTrue(permitted(valid, 0.5, 0.4, {"outside_ceiling": cap}))
        self.assertFalse(permitted(control, 0.5, 0.4, {"outside_ceiling": cap}))
        self.assertFalse(permitted(dict(valid, veto=True), 0.5, 0.4, {"outside_ceiling": cap}))

    def test_d6_scope_preserves_whole_ctc_phones_at_both_edges(self):
        greedy = np.array([1, 1, 2, 2])
        self.assertEqual(exterior_phone_counts(greedy, 1, 3), (0, 0))
        # A blank separates a repeated symbol into a different phone.
        greedy = np.array([1, 0, 1, 2, 2, 0, 2])
        self.assertEqual(exterior_phone_counts(greedy, 2, 4), (1, 1))

    def test_compact_d6_cutoff_uses_only_eligible_development_controls(self):
        def score(probability, veto=False):
            return {"d6_probability": probability, "veto": veto}

        rows = [(True, "human", score(0.97)), (False, "human", score(0.99, veto=True)),
                (False, "synthetic", score(0.35)), (False, "CHM150", score(0.60))]
        cutoff = calibrate_compact_d6(rows)
        self.assertEqual(cutoff, 0.60)
        self.assertFalse(compact_accepted(score(0.60), cutoff))
        self.assertTrue(compact_accepted(score(0.61), cutoff))
        self.assertFalse(compact_accepted(score(0.99, veto=True), cutoff))

    def test_compact_d6_context_keeps_human_order_and_rejects_quoted_trigger(self):
        def score(probability, cost, outside, veto=False):
            return {"d6_probability": probability, "veto": veto, "ctc_cost": cost, "outside": outside}

        order = score(0.71, 1.0, 2)
        quote = score(0.99, 0.1, 9)
        equally_short_control = score(0.65, 0.9, 1)
        rows = [(True, "human", order), (True, "synthetic", score(0.80, 0.1, 11)),
                (False, "human", quote), (False, "CHM150", equally_short_control)]
        limits = select_context_limits(rows)
        self.assertTrue(context_accepted(order, limits))
        self.assertFalse(context_accepted(quote, limits))
        self.assertFalse(context_accepted(equally_short_control, limits))
        self.assertFalse(context_accepted(dict(order, veto=True), limits))

    def test_adapted_bank_cannot_change_the_intent_head(self):
        with TemporaryDirectory() as work:
            directory = Path(work)
            reference, adapted = directory / "reference", directory / "adapted"
            reference.mkdir()
            adapted.mkdir()
            negative = directory / "index.json"
            negative.write_text("{}")
            weights = "student-int8-weights.npz"
            arrays = {"phones.weight": np.array([1], dtype=np.int8),
                      "phones.weight.scale": np.array([0.1], dtype=np.float32),
                      "phones.bias": np.array([1.0], dtype=np.float32),
                      "intent.weight": np.array([1], dtype=np.int8)}
            np.savez_compressed(reference / weights, **arrays)
            base_sha = file_digest(reference / weights)
            arrays["phones.weight"][0] += 1
            np.savez_compressed(adapted / weights, **arrays)
            args = SimpleNamespace(reference_bank=reference, base=adapted, negatives=directory)
            report = {"procedure": ADAPTATION, "initial_weights_sha256": base_sha,
                      "teacher_cache_sha256": "synthetic", "target_sha256": TARGET_SHA256,
                      "negative_cache_sha256": file_digest(negative), "weights_sha256": file_digest(adapted / weights)}
            with (patch("phoneme_confirm_lab.BASE_SHA256", base_sha),
                  patch("phoneme_confirm_lab.NEGATIVE_CACHE_SHA256", file_digest(negative))):
                self.assertEqual(checked_bank(args, report, "synthetic"), report["weights_sha256"])
                arrays["intent.weight"][0] += 1
                np.savez_compressed(adapted / weights, **arrays)
                report["weights_sha256"] = file_digest(adapted / weights)
                with self.assertRaisesRegex(ValueError, "changed a frozen tensor: intent.weight"):
                    checked_bank(args, report, "synthetic")


if __name__ == "__main__":
    unittest.main()
