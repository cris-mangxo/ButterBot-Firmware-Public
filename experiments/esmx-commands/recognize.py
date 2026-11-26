"""Desktop-only transcription and closed command routing for the robot fork."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unicodedata
import wave

SAMPLE_RATE = 16000
MAX_SECONDS = 8
MAX_TRANSCRIPT_BYTES = 128


def validate_wav(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("Se requiere WAV PCM mono, 16 bits, 16 kHz")
        frames = audio.getnframes()
        if not SAMPLE_RATE // 4 <= frames <= SAMPLE_RATE * MAX_SECONDS:
            raise ValueError("Duración admitida: 0.25 a 8 segundos")
        if len(audio.readframes(frames)) != frames * 2:
            raise ValueError("WAV PCM truncado")


def transcript_from_json(payload):
    segments = payload["transcription"]
    if not isinstance(segments, list):
        raise ValueError("Transcripción inválida")
    # Do not select an actionable fragment from a multi-segment turn.
    if len(segments) != 1:
        return ""
    text = segments[0]["text"]
    if not isinstance(text, str):
        raise ValueError("Segmento de voz inválido")
    return text


def transcribe(path, model, whisper):
    validate_wav(path)
    with tempfile.TemporaryDirectory(prefix="esmx-asr-") as folder:
        output = Path(folder) / "recognition"
        subprocess.run(
            [whisper, "-m", str(model), "-f", str(path), "-l", "es", "-t", "2",
             "-ng", "-nf", "-np", "-nt", "-oj", "-of", str(output)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        return transcript_from_json(json.loads(output.with_suffix(".json").read_text("utf-8")))


def route(router, transcript):
    if "\0" in transcript or len(transcript.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
        return {"status": "rejected"}
    checked_text = unicodedata.normalize("NFC", transcript)
    result = subprocess.run([str(router)], input=checked_text, check=True,
                            capture_output=True, text=True, encoding="utf-8", timeout=5)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--text")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--whisper", default="whisper-cli")
    args = parser.parse_args()

    if args.wav is not None and args.model is None:
        parser.error("--wav requiere --model")
    try:
        text = args.text
        if args.wav is not None:
            text = transcribe(args.wav.resolve(strict=True), args.model.resolve(strict=True), args.whisper)
        decision = route(args.router.resolve(strict=True), text)
    except (OSError, EOFError, ValueError, TypeError, KeyError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"ERROR DE PROTOTIPO ({type(error).__name__})\n")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
