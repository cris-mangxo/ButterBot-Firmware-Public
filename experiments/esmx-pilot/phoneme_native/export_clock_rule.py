"""Export the frozen strict clock contrast as a local C++ header."""
from argparse import ArgumentParser
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
import re


PHONE_COUNT = 32
MAX_TEMPLATE_PHONES = 32


def checked_path(tokens):
    if not 0 < len(tokens) <= MAX_TEMPLATE_PHONES or any(
        type(token) is not int or not 0 < token < PHONE_COUNT for token in tokens
    ):
        raise ValueError("Clock template is outside the native phoneme geometry")
    return ", ".join(map(str, tokens))


def write(out, recipe, arm):
    spec = json.loads(recipe.read_text())
    weight = spec["pins"]["weights"]
    if not re.fullmatch(r"[a-f0-9]{64}", weight):
        raise ValueError("Invalid weight fingerprint")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", arm) or arm not in spec["rules"]:
        raise ValueError("Clock arm is not in the frozen recipe")
    rule = spec["rules"][arm]["rule"]
    coefficients = rule["coefficient"]
    if len(coefficients) != 9 or not all(isfinite(value) for value in (*coefficients, rule["intercept"], rule["threshold"])):
        raise ValueError("Invalid clock contrast parameters")
    prefix = spec["forms"]["prefix"]
    if len(prefix) != 2:
        raise ValueError("The contrast requires exactly two prefixes")
    for identifier, tokens in zip(("EventPrefix", "NowPrefix"), prefix, strict=True):
        checked_path(tokens)
    banks = {
        name: [checked_path(tokens) for tokens in spec["forms"][name]]
        for name in ("now", "event", "other", "noun", "local_now")
    }
    for name, items in banks.items():
        if not items:
            raise ValueError(f"Missing templates for {name}")
    checksum = sha256(recipe.read_bytes()).hexdigest()
    with out.open("x") as handle:
        handle.write('#pragma once\n#include "ClockGuard.h"\n\nnamespace EsMx::PhonemeNative {\n')
        handle.write(f"static_assert(PhoneCount == {PHONE_COUNT} && MaxTemplatePhones == {MAX_TEMPLATE_PHONES});\n")
        handle.write(f'inline constexpr char ClockArm[] = "{arm}";\n')
        handle.write(f'inline constexpr char ClockBankSha256[] = "{weight}";\n')
        handle.write(f'inline constexpr char ClockRecipeSha256[] = "{checksum}";\n')
        for name, items in banks.items():
            ids = []
            for index, tokens in enumerate(items):
                identifier = f"Clock_{name}_{index}"
                ids.append(identifier)
                handle.write(f"inline constexpr std::uint8_t {identifier}[]{{{tokens}}};\n")
            handle.write(f"inline constexpr std::span<const std::uint8_t> Clock_{name}[]{{{', '.join(ids)}}};\n")
        for identifier, tokens in zip(("EventPrefix", "NowPrefix"), prefix, strict=True):
            handle.write(f"inline constexpr std::uint8_t {identifier}[]{{{checked_path(tokens)}}};\n")
        handle.write("inline constexpr ClockRule FrozenClock{\n")
        handle.write("    Clock_now, Clock_event, Clock_other, Clock_noun, Clock_local_now, EventPrefix, NowPrefix,\n")
        handle.write("    {" + ", ".join(map(repr, coefficients)) + "},\n")
        handle.write(f"    {rule['intercept']!r}, {rule['threshold']!r}\n}};\n}}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--arm", required=True, help="Explicit arm from the frozen recipe")
    args = parser.parse_args()
    write(args.out, args.recipe, args.arm)
