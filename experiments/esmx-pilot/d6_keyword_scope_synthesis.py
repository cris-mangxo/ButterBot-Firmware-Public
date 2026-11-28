"""Render an independent synthetic comparison panel in two Mexican Mac voices."""

import json
from pathlib import Path
import argparse

from keyword_lab import render
from time_lab import file_digest

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("output", type=Path)
r = cli.parse_args().output
if r.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
    raise ValueError("Keep generated audio outside the repository")
r.mkdir(mode=0o700)
voices=(("Grandma (Español (México))", 168),("Reed (Español (México))", 181))
pairs=(
    ("Tire un dado de seis por favor.", "No tire usted el dado de seis."),
    ("Échame un dado de seis, porfa.", "No me tire usted un dado de seis."),
    ("Lanza el dado de seis, por favor.", "Lanza el dado de seis, mejor no."),
    ("Oiga, aviente un dado de seis.", "Aviente un dado de seis. No, olvídalo."),
    ("Un dado de seis, aviéntalo.", "Si quiere, tire un dado de seis para cada jugador."),
    ("Me lanzas un dado de seis, porfa.", "No lance un dado de seis, por favor."),
)
out=[]
for index, pair in enumerate(pairs):
    for label,phrase in enumerate(pair):
        for id_voice,(voice,rate) in enumerate(voices):
            path=r/f"target-{index:02}-{label}-{id_voice}.wav"
            render(r,path.stem,voice,phrase,rate)
            out.append({"file":path.name,"voice_group":id_voice,"expected":6 if label==0 else 0,
                        "sha256":file_digest(path),"phrase":phrase})
(r/"seal.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
