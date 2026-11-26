#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/teams/HoulabData_1788502323/liucunyu/foldnar/structrfm_official_v0.0.8}
UPSTREAM=${UPSTREAM:-/home/rna_jev_structrfm_interface_upstream/mxfold2-v0.1.2}
PYBIND=${PYBIND:-/home/rna_jev_structrfm_interface_upstream/pybind11-v2.13.6}
PYTHON=${PYTHON:-/root/miniconda3/envs/lucaone_cu128/bin/python}
EXPECTED_UPSTREAM=51b213676708bebd664f0c40873a46e09353e1ee
EXPECTED_SHA256=616ec59fd9c853bebbcb811eeff8af8ec2ce2b7380971dda9de63b9cc33cb6d0
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "$EXPECTED_UPSTREAM"
test -z "$(git -C "$UPSTREAM" status --porcelain)"
test "$(git -C "$PYBIND" rev-parse HEAD)" = "$(git -C "$PYBIND" rev-parse v2.13.6^{})"
SOURCE="$UPSTREAM/mxfold2/src"
BUILD="$ROOT/build/mxfold2-v0.1.2-cp310"
RUNTIME="$ROOT/runtime/cp310"
mkdir -p "$BUILD" "$RUNTIME"
ln -sfn "$PYBIND" "$SOURCE/pybind11"
trap 'rm -f "$SOURCE/pybind11"' EXIT
cmake -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$PYTHON"
cmake --build "$BUILD" --parallel 2
ARTIFACT="$RUNTIME/interface.cpython-310-x86_64-linux-gnu.so"
cp "$BUILD/interface.cpython-310-x86_64-linux-gnu.so" "$ARTIFACT"
printf '%s  %s\n' "$EXPECTED_SHA256" "$ARTIFACT" | sha256sum -c -
test ! -e "$ROOT/reports/differential/differential_report.json" || "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS_OFFICIAL_CP38_VS_REBUILT_CP310" and d["candidate_interface_sha256"] == sys.argv[2]' "$ROOT/reports/differential/differential_report.json" "$EXPECTED_SHA256"
