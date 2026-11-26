import unittest

from phrases_human import group, select


class PublicSpeechSelectionTest(unittest.TestCase):
    def test_lexical_controls_remain_distinct(self):
        self.assertEqual(group({"no", "tengo", "seis", "fichas"}), "negated_near")
        self.assertEqual(group({"dado", "de", "seis", "colores"}), "literal_confuser")
        self.assertEqual(group({"nunca", "olvido", "ese", "dato"}), "negated_near")
        self.assertEqual(group({"pasa", "el", "dado"}), "other_dice_terms")

    def test_fresh_cohort_excludes_prior_speakers_and_text(self):
        rows = [
            {"path": f"{i:03d}.mp3", "client_id": str(i), "sentence": f"Este es el clip {i}"}
            for i in range(55)
        ]
        rows.extend([
            {"path": "998.mp3", "client_id": "0", "sentence": "Quince pasos completos"},
            {"path": "999.mp3", "client_id": "99", "sentence": "Este es el clip 0"},
        ])
        earlier, _, _ = select(rows, fresh=False)
        later, _, _ = select(rows, fresh=True)
        self.assertEqual(len(earlier), 24)
        self.assertEqual(len(later), 24)
        self.assertTrue(earlier.keys().isdisjoint(later))
        self.assertNotIn("998.mp3", later)
        self.assertNotIn("999.mp3", later)


if __name__ == "__main__":
    unittest.main()
