"""Piloto local de micrófono de la Mac. Genera una sesión privada por persona."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from threading import Timer
from uuid import uuid4
import wave

PRIVATE = Path.home() / "Library" / "Application Support" / "ButterBotPrivate"
DURATION = 4
PROMPTS = Path(__file__).with_name("prompts.json")


def check_wav(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("Formato WAV incorrecto")
        frames = audio.getnframes()
        if frames != 16000 * DURATION or len(audio.readframes(frames)) != frames * 2:
            raise ValueError("La toma no duró cuatro segundos completos")
    return frames


def record_wav(mic, wav):
    # The operator triggers every take. The progress timestamp is generated after input opens.
    with subprocess.Popen(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
         "-f", "avfoundation", "-i", f":{mic}", "-t", str(DURATION),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         "-progress", "pipe:1", "-stats_period", "0.1", str(wav)],
        stdout=subprocess.PIPE, text=True,
    ) as process:
        deadline = Timer(14, process.kill)
        deadline.start()
        try:
            for line in process.stdout:
                if line.startswith("out_time_us=") and line.strip() != "out_time_us=N/A":
                    if int(line.split("=", 1)[1]) > 0:
                        print("AHORA", flush=True)
                        break
            process.communicate(timeout=12)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
            raise
        finally:
            deadline.cancel()
        if process.returncode:
            raise ValueError("FFmpeg no pudo grabar. Revisa el micrófono y su permiso.")
    return check_wav(wav)


def save_metadata(session, metadata):
    update = session / "session.tmp"
    stream = update.open("x", encoding="utf-8")
    try:
        with stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        update.replace(session / "session.json")
    finally:
        update.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", required=True, help="Seudónimo, por ejemplo P01")
    parser.add_argument("--mic", required=True, type=int, help="Índice AVFoundation de audio")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--takes", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not re.fullmatch(r"P[0-9]{2}", args.person) or not 0 <= args.mic <= 99:
        parser.error("Usa un seudónimo P01, P02… y un índice de audio de 0 a 99")

    print("La persona que habla puede omitir una toma o detener la sesión. Los WAV quedan en esta Mac.")
    if input("Si eres la persona que hablará, escribe ACEPTO para registrar tu permiso local: ").strip() != "ACEPTO":
        return
    os.umask(0o077)
    # This fixed location is outside the worktrees and the repository.
    corpus = PRIVATE / "asr-piloto"
    if corpus.resolve() != corpus or corpus.is_symlink():
        parser.error("La carpeta privada no puede estar redirigida")
    corpus.mkdir(parents=True, exist_ok=True, mode=0o700)
    if corpus.stat().st_mode & 0o077:
        parser.error("La carpeta de sesiones debe ser de acceso exclusivo de su dueño")
    stamp = datetime.now().astimezone()
    session_id = f"{stamp:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    session = corpus / session_id
    session.mkdir(mode=0o700)
    metadata = {
        "schema":"butterbot-kws-pilot/1", "person_ref":args.person, "session_id":session_id,
        "partition":args.split, "day":stamp.date().isoformat(), "mic":"mac-avfoundation",
        "consent":"operator typed ACEPTO for private local project use", "entries":[],
    }
    save_metadata(session, metadata)
    prompts = json.loads(PROMPTS.read_text("utf-8"))
    try:
        for repeat in range(args.takes):
            for index, (label, sentence) in enumerate(prompts, start=1):
                code = f"{repeat + 1:02}-{index:02}"
                display = sentence if sentence is not None else "Silencio, sin hablar"
                print(f"\nToma {code}/ {display}")
                action = input("Enter para grabar, o para omitir, q para terminar: ").strip().lower()
                if action == "q":
                    return
                if action:
                    continue
                wav = session / f"{code}.wav"
                temporary = session / f"{code}.partial.wav"
                try:
                    frames = record_wav(args.mic, temporary)
                    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
                    temporary.rename(wav)
                    entry = {
                        "file":wav.name, "prompt_index":index, "label":label,
                        "sha256":checksum, "frames":frames,
                    }
                    updated = dict(metadata, entries=[*metadata["entries"], entry])
                    save_metadata(session, updated)
                    metadata = updated
                    print("Guardada")
                except (ValueError, OSError, wave.Error, subprocess.TimeoutExpired) as error:
                    print(f"Toma omitida por error de captura ({type(error).__name__}).")
                    if not metadata["entries"]:
                        return
                finally:
                    temporary.unlink(missing_ok=True)
    except (EOFError, KeyboardInterrupt):
        print("\nSesión detenida.")
    finally:
        print(json.dumps({"estado":"sesion-local", "particion":args.split,
                          "grabaciones":len(metadata["entries"]), "previstas":len(prompts)*args.takes}))


if __name__ == "__main__":
    main()
