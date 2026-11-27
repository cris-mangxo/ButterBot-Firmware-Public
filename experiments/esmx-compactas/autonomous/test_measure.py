import unittest

from measure import summarize


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
