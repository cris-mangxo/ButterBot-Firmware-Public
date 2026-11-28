"""Exercise competition and abstention in the complete-turn grammar."""
import unittest

import torch

from paired_phone_grammar_lab import Grammar, proposals


class GrammarTest(unittest.TestCase):
    def test_order_competes_with_other_intents(self):
        grammar = Grammar({"1": [[(1, 2)], [(4, 1, 2)]],
                           "5": [[(2, 3)], [(4, 2, 3)]],
                           "6": [[(3, 4)], [(2, 3, 4)]]})
        sequence = torch.tensor([0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0])
        logits = torch.full((len(sequence), 5), -5.0)
        logits.scatter_(1, sequence[:, None], 5.0)
        scores = grammar.scores(logits)
        self.assertGreater(scores["1"]["margin"], 0)
        for label in ("5", "6"):
            self.assertLess(scores[label]["margin"], 0)

    def test_path_with_adjacent_repetitions_needs_blanks(self):
        grammar = Grammar({"1": [[(1,) * 7], [(4,)]],
                           "5": [[(2,)], [(4,)]],
                           "6": [[(3,)], [(4,)]]})
        scores = grammar.scores(torch.zeros((12, 5)))
        self.assertNotIn("1", scores)
        self.assertEqual(grammar.scores(torch.full((12, 5), float("nan"))), {})

    def test_exact_threshold_and_multiple_passes_abstain(self):
        thresholds = {"bank": {"1": {"margin": 0.2}, "5": {"margin": 0.2}}}
        scores = {"bank": {"1": {"margin": 0.2}, "5": {"margin": 0.3}}}
        self.assertEqual(proposals(scores, thresholds, "bank", "margin"), 5)
        scores["bank"]["1"]["margin"] = 0.3
        self.assertEqual(proposals(scores, thresholds, "bank", "margin"), 0)


if __name__ == "__main__":
    unittest.main()
