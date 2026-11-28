"""Export the sealed Chiste laboratory rule without storing its inputs in Git."""

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path


GROUPS = ("word", "request", "denial", "positive", "negative")
NATIVE_SCORE_MARGIN = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("header", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    raw = args.recipe.read_bytes()
    if sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Joke rule does not match its verified recipe")
    data = json.loads(raw)
    if data["selected"] != "contrast" or data["transfer_positive_fitting"]:
        raise ValueError("Expected the contrast rule with synthetic fitting")
    rule = data["development"]["contrast"]["rule"]
    coefficients = rule["coefficient"]
    if len(coefficients) != 9 or not all(math.isfinite(x) for x in
                                       [*coefficients, rule["intercept"], rule["threshold"]]):
        raise ValueError("Invalid coefficient or threshold")
    threshold = rule["threshold"] + NATIVE_SCORE_MARGIN
    weights = data["seals"]["weights"]
    if len(weights) != 64 or any(char not in "0123456789abcdef" for char in weights):
        raise ValueError("Invalid bank digest")
    with args.header.open("x") as header:
        header.write('#pragma once\n#include "JokeGuard.h"\n\nnamespace EsMx::PhonemeNative {\n')
        header.write(f'inline constexpr char JokeBankSha256[] = "{weights}";\n')
        header.write(f'inline constexpr char JokeRecipeSha256[] = "{args.sha256}";\n')
        for group in GROUPS:
            paths = data["paths"][group]
            if not paths or len(paths) > 32:
                raise ValueError("Invalid CTC group length")
            names = []
            for index, path in enumerate(paths):
                if not path or len(path) > 32 or any(type(token) is not int or not 1 <= token < 32 for token in path):
                    raise ValueError("Invalid CTC token sequence")
                name = f"Joke_{group}_{index}"
                names.append(name)
                tokens = ", ".join(map(str, path))
                header.write(f'inline constexpr std::uint8_t {name}[] {{{tokens}}};\n')
            header.write(f'inline constexpr TokenPath Joke_{group}[] {{{", ".join(names)}}};\n')
        values = ", ".join(format(value, ".17g") for value in coefficients)
        header.write('inline constexpr JokeRule FrozenJoke{\n'
                     f'    {", ".join("Joke_" + key for key in GROUPS)},\n'
                     f'    {{{values}}}, {rule["intercept"]:.17g}, {threshold:.17g},\n'
                     '};\n}\n')


if __name__ == "__main__":
    main()
