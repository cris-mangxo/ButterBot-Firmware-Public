"""Verify the bridge and match host results against the actual Xtensa factory table."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from elftools.elf.elffile import ELFFile

from recognize import route

TABLE_SECTION = ".rodata._ZN12_GLOBAL__N_1L8MappingsE"
FACTORY_RELOCATION = re.compile(
    r"^(\w+)\s+R_XTENSA_32\s+Routine\* makeRoutine<([A-Za-z0-9_]+)>\(",
    re.MULTILINE,
)


def read_xtensa_mappings(dispatch_object, objdump):
    with dispatch_object.open("rb") as source:
        elf = ELFFile(source)
        if (elf.elfclass, elf.little_endian, elf["e_machine"]) != (32, True, "EM_XTENSA"):
            raise ValueError("Se requiere el objeto Xtensa LE de 32 bits")
        table = elf.get_section_by_name(TABLE_SECTION).data()

    reloc = subprocess.check_output([objdump, "-r", "-C", dispatch_object], text=True)
    section = reloc.split(f"RELOCATION RECORDS FOR [{TABLE_SECTION}]:\n", 1)[1].split("\n\n", 1)[0]
    entries = [(int(offset, 16), factory) for offset, factory in FACTORY_RELOCATION.findall(section)]
    # Xtensa GCC layout: 4-byte Scenario, 1-byte payload, padding, 4-byte factory.
    if not entries or len(table) != len(entries) * 12:
        raise ValueError("Tamaño o relocaciones inesperados en la tabla de CircuitMess")
    result = {}
    for index, (offset, factory) in enumerate(entries):
        start = index * 12
        if offset != start + 8:
            raise ValueError("Desplazamiento inesperado de la fábrica")
        key = (int.from_bytes(table[start:start + 4], "little"), table[start + 4])
        if key in result:
            raise ValueError("Ruta duplicada en la tabla")
        result[key] = factory
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--objdump", required=True, type=Path)
    args = parser.parse_args()

    objects = args.build / "esp-idf/main/CMakeFiles/__idf_main.dir/src"
    original = objects / "States/ScenarioRoutineMappings.cpp.obj"
    bridge = objects / "Audio/EsMxRoutine.cpp.obj"
    native_map = read_xtensa_mappings(original, args.objdump)
    relocated = subprocess.check_output([args.objdump, "-r", "-C", bridge], text=True)
    for symbol in ("EsMx::matchCommand", "routineForScenario"):
        if not re.search(r"^\w+\s+R_XTENSA_\w+\s+" + re.escape(symbol) + r"\(", relocated, re.MULTILINE):
            raise ValueError(f"Falta la referencia compilada a {symbol}")

    fixture = json.loads(Path(__file__).with_name("cases.json").read_text("utf-8"))
    routes = {}
    router = args.router.resolve(strict=True)
    for _, text, intent in fixture["audio"] + fixture["extra_text"]:
        if intent is None:
            continue
        match = route(router, text)
        key = match["scenario"], match["data"]
        factory = native_map[key]
        if match["intent"] != intent or factory != fixture["routes"][intent]:
            raise ValueError(f"La orden {intent} no conserva su rutina")
        routes[intent] = {"scenario": key[0], "data": key[1], "factory": factory}
    if set(routes) != set(fixture["routes"]):
        raise ValueError("Falta verificar una ruta")
    print(json.dumps({"original_factory_entries": len(native_map), "bridge_references_original_dispatch": True,
                      "verified_routes": routes}, indent=2))


if __name__ == "__main__":
    main()
