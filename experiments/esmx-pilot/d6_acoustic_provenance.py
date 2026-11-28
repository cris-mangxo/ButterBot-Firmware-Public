"""Fingerprint the desktop replay inputs tied to the reviewed clock profile."""

import json

from time_lab import file_digest


R32_SHA = "f25ae6a948028c304c23d9e500d29285c087bc79"
R32_EVIDENCE_SHA = "d5c69d7af6106e0a2970f95549f99d5055153187e9a125879c48e683fa259b95"
MODEL_FILES = ("mdef", "means", "variances", "mixture_weights",
               "transition_matrices", "feat.params", "noisedict")


def model_hashes(model, dictionary):
    return {name: file_digest(model / name) for name in MODEL_FILES} | {
        "dictionary": file_digest(dictionary),
    }


def clock_reference(evidence, model, host, capture=None):
    evidence_sha = file_digest(evidence)
    if evidence_sha != R32_EVIDENCE_SHA:
        raise ValueError("Expected the published R32 evidence")
    reviewed = json.loads(evidence.read_text())
    files = model_hashes(model, model / "clock.dict")
    pins = reviewed["acoustic_and_build_pins"]
    expected = pins["acoustic_model_sha256"] | {"dictionary": pins["dictionary_sha256"]}
    binary = {name: file_digest(path) for name, path in (("host", host), ("capture", capture)) if path}
    if (files != expected or reviewed["reviewed_sha"] != R32_SHA
            or any(fingerprint != reviewed["mac"]["host_binary_sha256"][name]
                   for name, fingerprint in binary.items())):
        raise ValueError("Clock model or executable differs from the reviewed R32 profile")
    return {"files": files, "binary": binary, "firmware_sha": R32_SHA,
            "evidence_sha256": evidence_sha, "heroico_panel_sha256": reviewed["mac"]["heroico_panel_sha256"]}
