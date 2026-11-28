"""Validate that a known transcript bounds a single development word."""

import unittest

from mfa_cv27_compare import target_span


class TargetSpanTest(unittest.TestCase):
    def setUp(self):
        self.entries = [(0, 0.2, "tira"), (0.2, 0.6, "dado"), (0.6, 0.9, "ahora")]

    def span(self, entries, transcript="tira dado ahora", samples=24_000):
        tiers = {"words": {"entries": entries}}
        return target_span(tiers, "dado", transcript.split(), samples)

    def test_extracts_one_attributed_interval(self):
        self.assertEqual(self.span(self.entries), ((3200, 9600), "development_only"))

    def test_rejects_missing_word(self):
        self.assertEqual(self.span(self.entries[:1]), (None, "target_missing_or_repeated"))

    def test_rejects_repeated_aligned_word(self):
        entries = self.entries + [(0.9, 1.0, "dado")]
        self.assertEqual(self.span(entries), (None, "target_missing_or_repeated"))

    def test_rejects_repeated_transcript_word(self):
        self.assertEqual(self.span(self.entries, "tira dado ahora dado"), (None, "target_missing_or_repeated"))

    def test_rejects_unmatched_neighbor(self):
        entries = self.entries[:2] + [(0.6, 0.9, "mañana")]
        self.assertEqual(self.span(entries), (None, "neighbor_unmatched"))

    def test_rejects_read_beyond_audio(self):
        self.assertEqual(self.span(self.entries, samples=9500), (None, "bounds_or_duration"))

    def test_rejects_clip_shorter_than_90_ms(self):
        entries = self.entries[:1] + [(0.2, 0.28, "dado"), (0.28, 0.9, "ahora")]
        self.assertEqual(self.span(entries), (None, "bounds_or_duration"))

    def test_rejects_clip_longer_than_900_ms(self):
        entries = self.entries[:1] + [(0.2, 1.12, "dado"), (1.12, 1.3, "ahora")]
        self.assertEqual(self.span(entries), (None, "bounds_or_duration"))

    def test_accepts_word_at_end(self):
        self.assertEqual(self.span(self.entries[:2], "tira dado"), ((3200, 9600), "development_only"))


if __name__ == "__main__":
    unittest.main()
