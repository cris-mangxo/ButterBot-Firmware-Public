#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -Wno-unused-parameter -DBB_ESMX_PILOT \
	-I "$repo/experiments/esmx-pilot/dispatch_stubs" -I "$repo/experiments/esmx-pilot/capture_stubs" -I "$repo/experiments/esmx-commands/host" \
	-I "$repo/main/src" -I "$repo/components/ButterBot-Common/src" -I "$repo/components/CMF/src" \
	"$repo/experiments/esmx-pilot/test_dispatch.cpp" "$repo/experiments/esmx-pilot/dispatch_probe.cpp" \
	"$repo/main/src/Audio/EsMxPilot/Dispatch.cpp" \
	-o "$scratch/test_dispatch"
"$scratch/test_dispatch"
