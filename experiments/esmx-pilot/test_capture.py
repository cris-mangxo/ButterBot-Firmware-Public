#!/usr/bin/env python3
"""Exercise PilotCapture against host substitutes for the AFE and the RTOS."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true", help="enable ASan and UBSan")
    parser.add_argument("--repo", type=Path, help="path to the firmware checkout")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo = (args.repo or here.parents[1]).resolve()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-DBB_ESMX_PILOT"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]

    with tempfile.TemporaryDirectory(prefix="bb-pilot-capture-test-") as directory:
        executable = Path(directory) / "test_capture"
        subprocess.run([
            *compiler, *flags, "-I", str(here / "capture_stubs"), "-I", str(repo / "main/src"),
            str(repo / "main/src/Audio/EsMxKws/PilotCapture.cpp"), str(here / "test_capture.cpp"),
            "-o", str(executable),
        ], check=True, timeout=45)
        subprocess.run([str(executable)], check=True, timeout=10)


if __name__ == "__main__":
    main()
