"""Regression tests for public corpus selection and archive streaming."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import common_voice


def make_archive(path, bad_clip=False):
    header = (
        "client_id\tpath\tsentence\n"
        "speaker-1\ta.mp3\tHoy están abiertas las ventanas.\n"
        "speaker-1\tb.mp3\tEl mercado está lejos.\n"
        "speaker-2\tc.mp3\tQué hora es.\n"
        "speaker-2\td.mp3\tLa orquesta llegó al teatro.\n"
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        # Put audio ahead of its manifest to require a second streaming pass.
        for name, payload in (("es/clips/a.mp3", b"a"), ("es/clips/d.mp3", b"d"),
                              ("es/dev.tsv", b"not a test split"), ("es/test.tsv", header)):
            entry = tarfile.TarInfo(name)
            if bad_clip and name == "es/clips/a.mp3":
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/not-a-clip"
                archive.addfile(entry)
            else:
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))


class CommonVoiceTest(unittest.TestCase):
    def test_selection_is_unique_by_speaker_and_excludes_command_words(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "fixture.tar.gz"
            make_archive(archive)
            prefix, rows = common_voice.read_test_split(archive)
            self.assertEqual(str(prefix), "es")
            self.assertEqual(common_voice.select_examples(rows, 2), {"a.mp3", "d.mp3"})
            with self.assertRaises(ValueError):
                common_voice.select_examples(rows, 3)

    def test_two_model_counts_exclude_speaker_ids_and_transcripts(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "fixture.tar.gz"
            make_archive(archive)

            def decode(command, **kwargs):
                source = Path(command[command.index("-i") + 1])
                Path(command[-1]).write_bytes(source.read_bytes())

            def classify(program, model, wav):
                return {"label": 6 if model == "new" and wav.read_bytes() == b"d" else 0}

            with patch.object(common_voice.subprocess, "run", side_effect=decode), \
                 patch.object(common_voice, "score", side_effect=classify):
                receipt = common_voice.evaluate(archive, "fixture", {"baseline": "old", "candidate": "new"}, 2)
            self.assertEqual(receipt["outside_command_trials"], 2)
            self.assertEqual(receipt["distinct_speakers"], 2)
            self.assertEqual(receipt["results"]["baseline"]["rejected"], 2)
            self.assertEqual(receipt["results"]["candidate"]["false_accepts_by_label"], {"6": 1})
            self.assertNotIn("speaker-", str(receipt))
            self.assertNotIn("orquesta", str(receipt))

    def test_selected_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "fixture.tar.gz"
            make_archive(archive, bad_clip=True)
            with self.assertRaises(ValueError):
                common_voice.evaluate(archive, "fixture", {"baseline": "old", "candidate": "new"}, 2)


if __name__ == "__main__":
    unittest.main()
