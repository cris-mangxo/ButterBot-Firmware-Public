"""Verify function-preserving widening and native checkpoint geometry."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from phoneme_native.export_bank import read_bank
from phoneme_replay_lab import frozen_model
from phoneme_student_lab import RECIPE, Student, WIDTH, quantize
from wider_phone_student_lab import WIDE, widen


class WiderPhoneTest(unittest.TestCase):
    def test_widening_preserves_predictions_for_padded_frames(self):
        torch.manual_seed(921)
        base = Student(32).eval()
        inputs = torch.rand((2, 20, 31))
        lengths = torch.tensor([19, 31])
        with torch.inference_mode():
            actual = widen(base)(inputs, lengths)
            expected = base(inputs, lengths)
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference)
        self.assertEqual(widen(base).input.out_channels, WIDE)

    def test_native_export_pins_width_and_geometry(self):
        for width, expected_bytes in [(WIDTH, 9180), (WIDE, 50320)]:
            with self.subTest(width=width), TemporaryDirectory() as tmp:
                folder = Path(tmp)
                bank = folder / "student-int8-weights.npz"
                report_path = folder / "report.json"
                arrays, _ = quantize(Student(32, width=width))
                np.savez_compressed(bank, **arrays)
                report = {"schema": RECIPE, "labels": 32, "weights_sha256": sha256(bank.read_bytes()).hexdigest()}
                if width != WIDTH:
                    report["width"] = width
                report_path.write_text(json.dumps(report))
                checksum, _, size, geometry = read_bank(folder)
                self.assertEqual((checksum, size, geometry), (report["weights_sha256"], expected_bytes, width))
                self.assertEqual(frozen_model(folder)[0].input.out_channels, width)
                report["width"] = WIDE if width == WIDTH else WIDTH
                report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    read_bank(folder)


if __name__ == "__main__":
    unittest.main()
