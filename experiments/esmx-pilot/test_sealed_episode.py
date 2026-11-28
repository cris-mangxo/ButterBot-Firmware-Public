import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import sealed_episode as harness


class SealedEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.wav = self.root / "local-input.wav"
        with wave.open(str(self.wav), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x01\x00" * 4800)
        executable = Path(sys.executable).resolve()
        self.args = argparse.Namespace(
            manifest=self.root / "panel.json", clock_dir=None,
            **dict.fromkeys(harness.ARTIFACTS, executable),
        )
        self.case = {
            "wav": self.wav.name, "wav_sha256": harness.digest(self.wav),
            "role": "heldout", "locale": "es-MX", "expected": 5,
            "source_id": "example fixture", "source_group": "sample", "speaker_group": "speaker",
            "rights": "Test fixture only",
        }
        self.spec = {
            "version": 1, "source_sha": "0" * 40,
            "artifact_sha256": dict.fromkeys(harness.ARTIFACTS, harness.digest(executable)),
            "clock_files_sha256": {}, "adjustment": [], "episodes": [self.case],
        }

    def seal(self):
        content = json.dumps(self.spec).encode()
        self.args.manifest.write_bytes(content)
        self.args.manifest_sha256 = sha256(content).hexdigest()

    def test_wav_bytes_are_pinned_before_subprocesses(self):
        self.seal()
        _, _, _, [(role, label, pcm)] = harness.read_panel(self.args)
        self.assertEqual((role, label, len(pcm)), ("heldout", 5, 9600))
        with self.wav.open("ab") as stream:
            stream.write(b"changed")
        with patch.object(harness, "invoke") as subprocesses:
            with self.assertRaisesRegex(harness.PanelError, "^WAV format or digest mismatch$"):
                harness.read_panel(self.args)
        subprocesses.assert_not_called()

    def test_heldout_is_disjoint_from_adjustment_and_regression(self):
        entry = {"source_group": "different source", "speaker_group": self.case["speaker_group"]}
        for key in ("adjustment", "episodes"):
            row = dict(entry)
            if key == "episodes":
                row = {**self.case, **entry, "role": "regression"}
            self.spec[key].append(row)
            self.seal()
            with self.subTest(group_inventory=key), self.assertRaisesRegex(
                    harness.PanelError, "reserved group overlaps"):
                harness.read_panel(self.args)
            self.spec[key].pop()

    def test_classified_prefix_cannot_pass_as_abstention(self):
        prefix = {"reason": "classified", "observed": 0, "route": "none", "voiced_frames": 18,
                  "input_samples": 4800, "input_samples_sent": 4480}
        done = subprocess.CompletedProcess([], 0, json.dumps(prefix).encode(), b"")
        with patch.object(harness, "invoke", return_value=done) as child:
            report = harness.retained_view({"capture": self.args.capture}, {}, 0,
                                          b"\x01\x00" * 4800, self.root / "out.aac")
        self.assertEqual(report, {"status": "capture_inconclusive"})
        child.assert_called_once()

    def test_invalid_signal_is_not_verified_as_rejection(self):
        decision = b'{"intent":0,"disposition":"invalid_audio","allows_fallback":false}'
        responses = [subprocess.CompletedProcess([], 0, content, b"") for content in (b"0\n", decision)]
        programs = {"recognizer": self.args.recognizer, "routine": self.args.routine}
        with patch.object(harness, "invoke", side_effect=responses) as child:
            report = harness.episode_view(programs, {}, 0, b"\x00\x00" * 4800, self.root / "out.aac")
        self.assertEqual(report["status"], "invalid_audio")
        self.assertEqual([call.args[1] for call in child.call_args_list], [(), ("--decision",)])


if __name__ == "__main__":
    unittest.main()
