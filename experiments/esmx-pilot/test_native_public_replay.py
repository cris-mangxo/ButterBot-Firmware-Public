import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from phoneme_native import compare


class NativePublicReplayTest(unittest.TestCase):
    def test_past_reserve_is_opt_in_and_missing_capture_is_unevaluated(self):
        rows = [
            ("negative", "public", "no-command", 0, Path("negative.wav"), None),
            ("prior", "public_reserved_time", "time", 5, Path("prior.wav"), None),
            ("synthetic", "synthetic_aliases", "alias", 5, Path("synthetic.wav"), None),
        ]
        args = Namespace(public_root=Path("."), frontend=Path("frontend"), capture=Path("capture"),
                         pilot=Path("pilot"), coerll=None, include_past_reserves=False)
        pcm = b"\x01\x00" * 160
        native_row = {"case": "negative", "frames": 9, "intent": 0, "expected_intent": 0}
        with (patch.object(compare, "recordings", return_value=rows),
              patch.object(compare, "pcm_from", return_value=pcm),
              patch.object(compare, "invoke", side_effect=lambda cmd, *_: b"" if cmd == args.capture else b"0"),
              patch.object(compare, "native_frames", side_effect=lambda _, data: object() if data else None),
              patch.object(compare, "run", side_effect=lambda _, __, name, ___: dict(native_row, case=name)),
              patch.object(compare, "file_digest", return_value="sealed")):
            default = compare.public_replay(args, Path("student"), None)
            args.include_past_reserves = True
            expanded = compare.public_replay(args, Path("student"), None)

        self.assertEqual([row["case"] for row in default["rows"]], ["negative", "negative"])
        self.assertEqual([row["case"] for row in expanded["rows"]], ["negative", "negative", "prior", "prior"])
        self.assertNotEqual(default["panel_sha256"], expanded["panel_sha256"])
        raw, retained = default["rows"]
        self.assertEqual((raw["intent"], raw["oracle_intent"], raw["pilot_intent"]), (0, 0, 0))
        self.assertEqual((retained["intent"], retained["oracle_intent"], retained["pilot_intent"]),
                         (None, None, None))
