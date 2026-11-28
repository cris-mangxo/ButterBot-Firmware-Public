"""Prepare the optional D6 acoustic assets in a new local directory."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

from clock_model_pack import write_array

ORDERS = ('tira un dado de seis',
 'tira el dado de seis',
 'tirar un dado de seis',
 'tirar el dado de seis',
 'tírame un dado de seis',
 'tírame el dado de seis',
 'tíreme un dado de seis',
 'tíreme el dado de seis',
 'lanza un dado de seis',
 'lanza el dado de seis',
 'lanzar un dado de seis',
 'lanzar el dado de seis',
 'aviéntate un dado de seis',
 'aviéntate el dado de seis',
 'avienta un dado de seis',
 'avienta el dado de seis',
 'échame un dado de seis',
 'échame el dado de seis',
 'tire un dado de seis',
 'tire el dado de seis',
 'tienes que tirar un dado de seis',
 'puedes tirar un dado de seis',
 'me puedes tirar un dado de seis',
 'tirar un dado de seis caras',
 'tíreme un dado de seis por favor',
 'tira un dado de seis caras',
 'tírame un dado de seis por favor',
 'lanza un dado de seis lados',
 'tire un dado de seis por favor')

TAILS = ('por favor', 'porfa', 'seis caras', 'seis lados')

DENIALS = ('no tires un dado de seis',
 'no tire un dado de seis',
 'no tires el dado de seis',
 'no lances un dado de seis',
 'no avientes un dado de seis',
 'no quiero que tires un dado de seis',
 'no quiero tirar un dado de seis',
 'no me tires un dado de seis',
 'no me tire un dado de seis',
 'no vamos a tirar un dado de seis',
 'no hay que tirar un dado de seis',
 'mejor no tires un dado de seis',
 'mejor no tire un dado de seis',
 'tira un dado de seis mejor no',
 'tirar un dado de seis mejor no',
 'tire un dado de seis mejor no',
 'tira el dado de seis mejor no',
 'cancela la tirada de un dado de seis',
 'olvida la tirada de un dado de seis',
 'tira un dado de seis olvídalo',
 'tira un dado de seis por cada',
 'tirar un dado de seis por cada',
 'si necesitas tirar un dado de seis',
 'no tire usted un dado de seis',
 'no tire usted el dado de seis',
 'no me tire usted un dado de seis',
 'no tire usted un dado de seis por favor',
 'no lance un dado de seis',
 'no lance usted un dado de seis',
 'tira un dado de seis cancélalo',
 'aviente un dado de seis no olvídalo')

LATER_DENIALS = ('cada atributo',
 'cada estadística',
 'cancélalo',
 'cuántos dados',
 'dos dados',
 'déjalo',
 'mejor no',
 'mejor olvídalo',
 'no mejor no',
 'no olvídalo',
 'olvídalo',
 'por cada',
 'tres dados',
 'varios dados')

EARLIER_DENIALS = (
    "no tires", "no lances", "no avientes", "no tire", "ni tires",
    "no lo tires", "no quiero que", "cancela la tirada", "olvida la tirada",
    "si necesitas", "si necesito", "si quieres", "se tiran",
)

OTHER_DICE = tuple(f"un dado de {count}" for count in ("ocho", "diez", "doce", "veinte", "cien"))
NEGATIVE_GROUPS = (DENIALS, EARLIER_DENIALS, LATER_DENIALS, OTHER_DICE)
KEYWORDS_NEGATIVE = tuple(phrase for group in NEGATIVE_GROUPS for phrase in group)

COMMON = ('\n'
 '<emphasis> = [y | oye | oiga | por favor | porfa];\n'
 '<roll> = (tira | tirar | tire | tires | lance | lances | lanza | lanzar | tírame | tíreme |\n'
 '          avienta | aviente | avientes | aventar | echa | eche | échame | échate) [usted];\n'
 '<prelude> = [me | puedes | me puedes | podrías | tienes que | puedes tú];\n'
 '<d6> = (un dado | el dado | uno | un solo dado) de seis [caras | lados];\n'
 '<end> = [por favor | porfa];\n')

AFFIRMATIVE = '<emphasis> <prelude> <roll> <d6> <end>'

NEGATIVE = '<emphasis> (no | no no | mejor no | no quiero que | ni) <prelude> <roll> <d6> <end>'

PRONUNCIATIONS = {'avientes': 'a b i e_7 n t e s',
 'cancélalo': 'k a n s e_7 l a l o',
 'porfa': 'p o_7 r( f a',
 'tíreme': 't i_7 r( e m e'}

FIXES = {
    "usted": ("u_7 s t e d", "u s t e_7 d"),
    "lance": ("l a_7 n s", "l a_7 n s e"),
}
CLOCK_DICTIONARY_SHA = "68b218e5fbd2f381944cea5bf33b4267905bde66bec022a17d7a84ee1cf46ccf"


def grammar(tag, production):
    return f"#JSGF V1.0 UTF-8;\ngrammar contrast_{tag};\npublic <turn> = {production};\n{COMMON}".strip()


def generated_lexicon(source, clock):
    patterns = " ".join((*ORDERS, *TAILS, *KEYWORDS_NEGATIVE))
    paths = re.sub(r"<[^>]+>", " ", COMMON + AFFIRMATIVE + NEGATIVE)
    words = set(re.findall(r"[a-záéíóúñ]+", patterns + paths))
    with gzip.open(source, "rt", encoding="utf-8") as data:
        chosen = {line.split()[0]: " ".join(line.split()[1:]) for line in data
                  if line.strip() and line.split()[0] in words}
    for word, (expected, corrected) in FIXES.items():
        if chosen.get(word) != expected:
            raise ValueError(f"Unexpected imported MEXBET entry: {word}")
        chosen[word] = corrected
    for word in words - chosen.keys():
        if word not in PRONUNCIATIONS:
            raise ValueError(f"Missing imported MEXBET entry: {word}")
        chosen[word] = PRONUNCIATIONS[word]
    original = clock.read_bytes()
    if sha256(original).hexdigest() != CLOCK_DICTIONARY_SHA:
        raise ValueError("Expected the reviewed clock dictionary")
    present = {line.split()[0]: " ".join(line.split()[1:]) for line in original.decode().splitlines() if line.strip()}
    for word in chosen.keys() & present.keys():
        if chosen[word] != present[word]:
            raise ValueError(f"Overlapping dictionary pronunciation differs: {word}")
    additions = [f"{word} {chosen[word]}\n" for word in sorted(chosen.keys() - present.keys())]
    return original + b"\n" + "".join(additions).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("source", "clock", "output"):
        parser.add_argument("--" + item, type=Path, required=True)
    args = parser.parse_args()
    lexicon = generated_lexicon(args.source, args.clock)
    if len(KEYWORDS_NEGATIVE) != len(set(KEYWORDS_NEGATIVE)):
        raise ValueError("A negative keyword must belong to exactly one scope")
    keywords = "".join(f"{text} /1e-{floor}/\n" for group, floor in
                       ((ORDERS, 60), (TAILS, 24), (KEYWORDS_NEGATIVE, 56)) for text in group).encode()
    args.output.mkdir(mode=0o700)
    (args.output / "esmx-dice.dict").write_bytes(lexicon)
    (args.output / "dice.kws").write_bytes(keywords)
    with (args.output / "DiceAcousticAssets.h").open("x", encoding="utf-8") as out:
        out.write("#pragma once\n#include <cstdint>\nnamespace EsMx::Pilot::DiceAcousticAssets {\n")
        write_array(out, "Dictionary", lexicon)
        write_array(out, "Keywords", keywords)
        for name, values in (("Orders", ORDERS), ("Tails", TAILS), ("Denials", DENIALS),
                             ("EarlierDenials", EARLIER_DENIALS), ("LaterDenials", LATER_DENIALS),
                             ("OtherDice", OTHER_DICE)):
            out.write(f"inline constexpr const char* {name}[] = {{\n")
            out.writelines("    " + json.dumps(value, ensure_ascii=False) + ",\n" for value in values)
            out.write("};\n")
        for name, rule in (("Affirmative", AFFIRMATIVE), ("Negative", NEGATIVE)):
            out.write(f'inline constexpr char {name}[] = R"dice({grammar(name.lower(), rule)})dice";\n')
        out.write("}\n")


if __name__ == "__main__":
    main()
