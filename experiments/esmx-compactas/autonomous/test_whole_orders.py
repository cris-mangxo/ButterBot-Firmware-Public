import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import whole_orders
import public_orders
from public_orders import ESMX_FIXTURE, EXPANDED_FIXTURE, FIXTURE, HELDOUT_FIXTURES, NATURAL_FIXTURE, ORDERED_FIXTURE
from public_orders import specification as reserved, verify_unique_ids


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

    def test_expanded_heldout_group_remains_reserved_under_filename_alias(self):
        source = next(iter(reserved(EXPANDED_FIXTURE)["sources"].values()))
        data = whole_orders.specification(whole_orders.EXPANDED_FIXTURE)
        data["sources"]["apaga"]["speaker_group"] = source["speaker_group"]
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "training-copy.json"
            fixture.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "panel reservado"):
                whole_orders.specification(fixture)

    def test_composed_negation_sources_remain_reserved(self):
        panel = reserved(ORDERED_FIXTURE)
        prefixes = {case["negation_prefix"]["source"] for case in panel["cases"] if "negation_prefix" in case}
        self.assertEqual(len(prefixes), 2)
        for key in prefixes:
            with self.subTest(prefix=key):
                self.spec["sources"]["apaga"]["speaker_group"] = panel["sources"][key]["speaker_group"]
                with self.assertRaisesRegex(ValueError, "panel reservado"):
                    self.verify()

    def test_natural_panel_group_and_episode_remain_reserved(self):
        for source in reserved(NATURAL_FIXTURE)["sources"].values():
            for key in ("speaker_group", "enclosure", "sha256"):
                with self.subTest(source=source["speaker_group"], key=key):
                    meta = self.spec["sources"]["apaga"]
                    previous, meta[key] = meta[key], source[key]
                    try:
                        with self.assertRaisesRegex(ValueError, "panel reservado"):
                            self.verify()
                    finally:
                        meta[key] = previous

    def test_reused_id_from_another_panel_is_rejected(self):
        verify_unique_ids()
        panel = reserved(NATURAL_FIXTURE)
        panel["cases"][0]["id"] = reserved(FIXTURE)["cases"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "other-panel.json"
            other.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ID repetido"):
                verify_unique_ids((FIXTURE, other))

    def test_natural_panel_requires_digest_per_clip(self):
        panel = reserved(NATURAL_FIXTURE)
        del panel["cases"][0]["pcm_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "natural-copy.json"
            copy.write_text(json.dumps(panel), encoding="utf-8")
            with patch.object(public_orders, "NATURAL_FIXTURE", copy):
                with self.assertRaisesRegex(ValueError, "Falta la huella PCM"):
                    reserved(copy)

    def test_esmx_sources_are_disjoint_and_reserved_by_group_url_and_hash(self):
        sources = reserved(ESMX_FIXTURE)["sources"]
        others = [(name, meta) for fixture in HELDOUT_FIXTURES if fixture != ESMX_FIXTURE
                  for name, meta in reserved(fixture)["sources"].items()]
        groups = {meta.get("speaker_group", name) for name, meta in others}
        assets = {meta[key] for _, meta in others for key in ("enclosure", "sha256")}
        for source in sources.values():
            self.assertNotIn(source["speaker_group"], groups)
            for key in ("enclosure", "sha256"):
                self.assertNotIn(source[key], assets)
            for key in ("speaker_group", "enclosure", "sha256"):
                with self.subTest(source=source["speaker_group"], key=key):
                    meta = self.spec["sources"]["apaga"]
                    previous, meta[key] = meta[key], source[key]
                    try:
                        with self.assertRaisesRegex(ValueError, "panel reservado"):
                            self.verify()
                    finally:
                        meta[key] = previous

    def test_esmx_profile_requires_hash_after_renaming(self):
        panel = reserved(ESMX_FIXTURE)
        del panel["cases"][0]["pcm_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "renamed.json"
            copy.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Falta la huella PCM"):
                reserved(copy)

    def test_esmx_cancellation_requires_valid_intervals(self):
        panel = reserved(ESMX_FIXTURE)
        case = next(row for row in panel["cases"] if row["category"] == "contextual_single_d6_cancel")
        case["context_ms"] = [case["start_ms"] + 100, case["end_ms"]]
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "renamed.json"
            copy.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contexto de la toma"):
                reserved(copy)

    def test_secondary_controls_cannot_be_counted_as_esmx_orders(self):
        panel = reserved(ESMX_FIXTURE)
        case = next(row for row in panel["cases"] if row.get("supplemental_control"))
        case["expected"] = 6
        case["category"] = "explicit_single_d6_order"
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "renamed.json"
            copy.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controles externos marcados"):
                reserved(copy)
