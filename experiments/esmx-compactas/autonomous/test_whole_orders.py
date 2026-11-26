import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import whole_orders
from public_orders import specification as reserved


class WholeOrderSplitTest(unittest.TestCase):
    def setUp(self):
        self.spec = whole_orders.specification()

    def verify(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "whole_orders.json"
            fixture.write_text(json.dumps(self.spec), encoding="utf-8")
            with patch.object(whole_orders, "FIXTURE", fixture):
                return whole_orders.specification()

    def test_tempo_variants_remain_in_training(self):
        rows = list(whole_orders.order_rows(self.verify()))
        self.assertTrue(any(row["tempo"] != 100 and row["label"] == 1 for row in rows))
        self.assertTrue(all(row["tempo"] == 100 for row in rows if row["split"] == "dev"))

    def test_same_show_in_calibration_is_rejected(self):
        self.spec["sources"]["apaga"]["speaker_group"] = self.spec["sources"]["brainless"]["speaker_group"]
        with self.assertRaisesRegex(ValueError, "grupo de voces"):
            self.verify()

    def test_reserved_audio_with_renamed_source_is_rejected(self):
        first = next(iter(reserved()["sources"].values()))
        self.spec["sources"]["apaga"]["sha256"] = first["sha256"]
        with self.assertRaisesRegex(ValueError, "panel reservado"):
            self.verify()

    def test_overlapping_audio_with_same_source_is_rejected(self):
        duplicate = dict(self.spec["cases"][0], id="train-99")
        self.spec["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "superponen"):
            self.verify()
