import json
from pathlib import Path
import tempfile
import unittest

from ald_round import digest
from d6_guard import LOCAL_CV_SHA256
from human_words import REVISION, read_manifest


class HumanPartitionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        audio = self.folder / "audio"
        audio.mkdir()
        self.manifest = {
            "revision": REVISION, "reference_sha256": LOCAL_CV_SHA256,
            "reserved_common_voice_ids": 0, "eligible_word_counts": {}, "samples": [],
        }
        for split, speaker, source in (("train", "f" * 64, 1), ("dev", "0" * 64, 2)):
            name = f"{split}-0000"
            wav, annotation = audio / f"{name}.wav", audio / f"{name}.json"
            wav.write_bytes(b"fixture")
            annotation.write_text("{}", encoding="utf-8")
            self.manifest["samples"].append({
                "id": name, "split": split, "label": 1, "text": "tira",
                "sources": [[speaker, f"tira_common_voice_es_{source}.opus"]],
                "wav_sha256": digest(wav), "annotation_sha256": digest(annotation),
            })

    def verify(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        return read_manifest(self.folder)[1]

    def test_distinct_voices_and_clips(self):
        self.assertEqual(self.verify()["speakers"], {"train": 1, "dev": 1})

    def test_shared_original_clip_rejected_even_with_different_word(self):
        self.manifest["samples"][1]["sources"][0][1] = "seis_common_voice_es_1.opus"
        with self.assertRaisesRegex(ValueError, "grabaciones distintas"):
            self.verify()

    def test_shared_speaker_rejected(self):
        self.manifest["samples"][1]["sources"][0][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fuera de su partición"):
            self.verify()

    def test_changed_alignment_rejected(self):
        (self.folder / "audio" / "dev-0000.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cambió"):
            self.verify()
