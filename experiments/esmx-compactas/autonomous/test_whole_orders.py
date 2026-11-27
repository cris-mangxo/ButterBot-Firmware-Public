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

    def test_direct_esmx_panel_is_reserved_for_all_training_manifests(self):
        fixture = public_orders.DIRECT_ESMX_FIXTURE
        for source in reserved(fixture)["sources"].values():
            self.spec["sources"]["apaga"]["speaker_group"] = source["speaker_group"]
            with self.subTest(source=source["speaker_group"]):
                with self.assertRaisesRegex(ValueError, "panel reservado"):
                    self.verify()

    def test_native_training_rejects_reserved_voice_and_program_crossing(self):
        data = whole_orders.specification(whole_orders.ESMX_FIXTURE)
        development = data["sources"]["nayeli"]
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "training.json"
            bad_sources = (
                (dict(development, speaker_group="kill-team-mexico"), "panel reservado"),
                (dict(development, publisher="Trompos y más"), "programa aparece"),
            )
            for source, error in bad_sources:
                with self.subTest(error=error):
                    data["sources"]["nayeli"] = source
                    fixture.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        whole_orders.specification(fixture)

    def test_native_training_checks_evidence_within_context(self):
        data = whole_orders.specification(whole_orders.ESMX_FIXTURE)
        case = next(case for case in data["cases"] if case["category"] == "positive")
        case["die_evidence_ms"] = [0, 1]
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "training.json"
            fixture.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "un solo D6"):
                whole_orders.specification(fixture)

    def test_explicit_transfers_preserve_archived_takes(self):
        data = whole_orders.specification(whole_orders.EXPLICIT_ESMX_FIXTURE)
        commands = [case for case in data["cases"]
                    if case["id"].startswith("train-") and case["category"] == "positive"]
        self.assertEqual({case["archive_id"] for case in commands}, {"public-71", "public-72"})
        self.assertEqual({case["source"] for case in commands}, {"fungi", "juganor"})
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "explicit-training.json"
            with patch.object(whole_orders, "EXPLICIT_ESMX_FIXTURE", fixture):
                commands[0]["pcm_sha256"] = "0" * 64
                fixture.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "difiere de su caso archivado"):
                    whole_orders.specification(fixture)
                commands[0]["pcm_sha256"] = next(
                    case["pcm_sha256"] for case in reserved(public_orders.DIRECT_ESMX_FIXTURE)["cases"]
                    if case["id"] == commands[0]["archive_id"]
                )
                control = next(case for case in data["cases"] if case.get("archive_id") == "public-74")
                control["category"] = "outside"
                fixture.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Cambió la etiqueta"):
                    whole_orders.specification(fixture)

    def test_omission_keeps_the_decision_within_the_context(self):
        data = reserved(public_orders.CANCELLATION_FIXTURE)
        case, = data["cases"]
        self.assertEqual(case["expected"], 0)
        self.assertTrue(case["supplemental_control"])
        case["die_evidence_ms"] = [0, 1]
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "panel.json"
            fixture.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "omitir queda fuera del contexto"):
                reserved(fixture)

    def test_explicit_split_drops_reassigned_shows_and_composed_prefixes(self):
        from measure import is_reassigned, reassigned_assets
        data = whole_orders.specification(whole_orders.EXPLICIT_ESMX_FIXTURE)
        groups = whole_orders.explicit_exclusions(data["sources"])
        assets = reassigned_assets(groups)
        sources = {"tirando-rol": {}, "reroll": {"speaker_group": "reroll"},
                   "guest": {"speaker_group": "condado-veinte"},
                   "publisher-only": {"publisher": "Tirando Rol"},
                   "hash-only": {"sha256": data["sources"]["fungi"]["sha256"]}}
        for name in ("tirando-rol", "guest", "publisher-only", "hash-only"):
            self.assertTrue(is_reassigned({"source": name}, sources, groups, assets))
        self.assertTrue(is_reassigned({"source": "reroll",
                                       "negation_prefix": {"source": "tirando-rol"}}, sources, groups, assets))
        self.assertFalse(is_reassigned({"source": "reroll"}, sources, groups, assets))

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
