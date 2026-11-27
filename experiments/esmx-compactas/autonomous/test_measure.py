import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ald_round import digest
from measure import is_reassigned, summarize, verify_backup_training
from whole_orders import BACKUP_ESMX_FIXTURE, EXPLICIT_ESMX_FIXTURE


class MeasureCountsTest(unittest.TestCase):
    def test_overlong_turn_is_not_counted_as_correct_rejection(self):
        rows = [
            {"category": "outside", "expected": 0, "overlong": True, "decisions": {"model": 0}},
            {"category": "outside", "expected": 0, "overlong": False, "decisions": {"model": 0}},
            {"category": "outside", "expected": 0, "overlong": False, "decisions": {"model": 6}},
        ]
        result = summarize(rows)["outside"]
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["overlong"], 1)
        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["correct"]["model"], 1)
        self.assertEqual(result["false_d6"]["model"], 1)


class BackupFitTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "fit.json"
        self.bank_sha = "1" * 64
        self.fit = {
            "bank_sha256": self.bank_sha,
            "continuous": {
                "fixture_sha256": digest(EXPLICIT_ESMX_FIXTURE),
                "positive_filter": "dado de seis",
                "included_human_positive_ids": ["train-11", "train-12"],
            },
        }

    def verify(self, bank_sha):
        self.path.write_text(json.dumps(self.fit), encoding="utf-8")
        return verify_backup_training(self.path, bank_sha)

    def test_verified_backup_excludes_training_calibration_and_guest_groups(self):
        groups, assets = self.verify(self.bank_sha)
        self.assertTrue({"tirando-rol", "condado-veinte", "el-juganor", "tiempo-de-pensar"} <= groups)
        self.assertFalse({"reroll", "role-per-second"} & groups)
        self.assertTrue(is_reassigned({"source": "alias"}, {"alias": {
            "sha256": "98f4b462392b9124404d25e952a9df90f0d428db4bf2f6159a3302a440b21dab",
        }}, groups, assets))

    def test_rejects_bank_that_does_not_match(self):
        with self.assertRaisesRegex(ValueError, "no corresponde"):
            self.verify("2" * 64)

    def test_rejects_unknown_human_example(self):
        self.fit["continuous"]["included_human_positive_ids"].append("public-88")
        with self.assertRaisesRegex(ValueError, "no acredita"):
            self.verify(self.bank_sha)

    def test_expanded_fit_requires_the_new_literal_example(self):
        continuous = self.fit["continuous"]
        continuous["fixture_sha256"] = digest(BACKUP_ESMX_FIXTURE)
        with self.assertRaisesRegex(ValueError, "no acredita"):
            self.verify(self.bank_sha)
        continuous["included_human_positive_ids"].append("train-14")
        groups, assets = self.verify(self.bank_sha)
        self.assertIn("tirando-rol", groups)
        self.assertIn("bc00c16316407f2fbf87c51359d0a56a090ef173eec1d0ebe88e280bc74a3674", assets)

    def test_rejects_pooled_fit(self):
        self.fit["continuous"]["corpora"] = [{"fixture_sha256": digest(EXPLICIT_ESMX_FIXTURE)}]
        with self.assertRaisesRegex(ValueError, "requiere el corpus"):
            self.verify(self.bank_sha)
