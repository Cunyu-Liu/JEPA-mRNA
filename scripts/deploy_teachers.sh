#!/bin/bash
# Deploy + version-lock the System-2 teacher models (Task 11).
#
# The teacher ensemble is
#
#     p^teacher = mean(p^ViennaRNA, p^RNAstructure, p^LinearPartition)
#
# and its soft labels are only reproducible if the software versions *and* the
# Turner parameter set are locked, because the Turner parameters change between
# ViennaRNA releases (spec §3.5, §5.7, §8.1 G5).  This script therefore:
#
#   1. probes what is actually installed and records the detected versions;
#   2. reports what is missing, loudly and per tool;
#   3. prints the exact install commands for the pinned versions.
#
# **It does not install anything by default, and it never pretends to.**  This
# environment has no network access, so `--install` first probes connectivity and
# refuses with an actionable message when the probe fails; the install commands are
# printed instead.  Pinned versions default to `待核验` (to be filled by the
# operator) rather than being invented -- the spec forbids fabricating version
# numbers, and a wrong lock is worse than an explicit unknown.
#
# Usage:
#   deploy_teachers.sh [--check] [--commands] [--install] [--dry-run]
#                      [--lock-file PATH] [--vienna-version V]
#                      [--rnastructure-version V] [--linearpartition-version V]
#
#   # the normal path here (no network): report + commands, write the lock file
#   deploy_teachers.sh --check --commands
#
# Exit codes: 0 = every teacher present and locked, 4 = at least one teacher
# missing (the script is still a success in the sense that it reported honestly).
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
PY="${RNAJEPA_PYTHON:-python3}"
LOCK_FILE="${RNAJEPA_TEACHER_LOCK:-$ART/teachers/teacher_versions.json}"

# Pinned versions.  `待核验` means "not yet verified offline" -- fill these in from
# the vendor release pages before generating soft labels, then re-run --check.
VIENNA_VERSION="${RNAJEPA_VIENNA_VERSION:-待核验}"
RNASTRUCTURE_VERSION="${RNAJEPA_RNASTRUCTURE_VERSION:-待核验}"
LINEARPARTITION_VERSION="${RNAJEPA_LINEARPARTITION_VERSION:-待核验}"
TURNER_PARAMS_VERSION="${RNAJEPA_TURNER_PARAMS_VERSION:-待核验}"

DO_CHECK=0; DO_COMMANDS=0; DO_INSTALL=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) DO_CHECK=1; shift;;
    --commands) DO_COMMANDS=1; shift;;
    --install) DO_INSTALL=1; shift;;
    --dry-run) DRY=1; DO_CHECK=1; DO_COMMANDS=1; shift;;
    --lock-file) LOCK_FILE="$2"; shift 2;;
    --vienna-version) VIENNA_VERSION="$2"; shift 2;;
    --rnastructure-version) RNASTRUCTURE_VERSION="$2"; shift 2;;
    --linearpartition-version) LINEARPARTITION_VERSION="$2"; shift 2;;
    --turner-params-version) TURNER_PARAMS_VERSION="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ $((DO_CHECK + DO_COMMANDS + DO_INSTALL)) -eq 0 ] && DO_CHECK=1

say() { echo "[deploy_teachers] $*"; }

# ------------------------------------------------------------------ detection
# `detect <label> <exe> <version-flag> <python-module> <identity-regex>`
# Echoes "<status>|<version>|<path>"; status is "present", "unverified" or "missing".
#
# The identity regex guards against a same-named unrelated binary.  This is not
# hypothetical: on macOS `/usr/bin/Fold` is the BSD text-wrapper `fold`, not
# RNAstructure's folding program, and it happily exists while being useless here.
# A tool whose identity cannot be confirmed is reported as "unverified", never as
# "present" -- a false "present" would silently poison the soft-label corpus.
detect() {
  local label="$1" exe="$2" vflag="$3" module="$4" identity="$5"
  local path version="待核验" out
  if [ -n "$exe" ] && command -v "$exe" >/dev/null 2>&1; then
    path="$(command -v "$exe")"
    out=""
    if [ -n "$vflag" ]; then
      out="$( { "$exe" "$vflag" 2>&1 || true; } | head -3 | tr -d '\r' )"
    fi
    if [ -n "$out" ] && printf '%s' "$out" | grep -Eq "$identity"; then
      version="$(printf '%s' "$out" | head -1)"
      echo "present|$version|$path"
      return 0
    fi
    if printf '%s' "$path" | grep -Eqi "$identity"; then
      echo "present|$version|$path"
      return 0
    fi
    echo "unverified|待核验|$path (identity not confirmed by '$exe $vflag')"
    return 0
  fi
  if [ -n "$module" ]; then
    version="$("$PY" -c "
import importlib, sys
try:
    m = importlib.import_module('$module')
except Exception:
    sys.exit(1)
print(getattr(m, '__version__', '待核验'))
" 2>/dev/null)" && [ -n "$version" ] && { echo "present|$version|python:$module"; return 0; }
  fi
  echo "missing|待核验|"
}

VIENNA="$(detect viennarna RNAfold --version RNA 'RNAfold|ViennaRNA')"
RNASTR="$(detect rnastructure Fold --version '' 'RNAstructure')"
LINEARP="$(detect linearpartition linearpartition --version '' 'LinearPartition')"

# Turner parameter files shipped with ViennaRNA (version-sensitive).
TURNER_FILES=""
if [ "${VIENNA%%|*}" = "present" ]; then
  vienna_dir="$(dirname "${VIENNA##*|}")"
  for candidate in "$vienna_dir/../share/ViennaRNA" "$vienna_dir/../share/viennarna" \
                   /usr/local/share/ViennaRNA /usr/share/ViennaRNA \
                   /opt/homebrew/share/ViennaRNA; do
    if [ -d "$candidate" ]; then
      found="$(ls "$candidate" 2>/dev/null | grep -E '^rna_turner.*\.par$' | tr '\n' ' ')"
      [ -n "$found" ] && TURNER_FILES="$found" && break
    fi
  done
fi
[ -z "$TURNER_FILES" ] && TURNER_FILES="待核验"

# ------------------------------------------------------------------ reporting
report() {
  say "teacher deployment status (nothing is installed by this script by default)"
  printf '  %-16s %-11s %-30s %s\n' TOOL STATUS VERSION PATH
  local row
  for row in "viennarna|$VIENNA" "rnastructure|$RNASTR" "linearpartition|$LINEARP"; do
    local label="${row%%|*}" rest="${row#*|}"
    local status="${rest%%|*}" r="${rest#*|}"
    local version="${r%%|*}" path="${r#*|}"
    printf '  %-16s %-11s %-30s %s\n' "$label" "$status" "$version" "$path"
  done
  printf '  %-16s %-11s %-30s %s\n' "turner_params" "n/a" "$TURNER_PARAMS_VERSION" "$TURNER_FILES"
  echo
  local missing=0 unverified=0
  for row in "$VIENNA" "$RNASTR" "$LINEARP"; do
    case "${row%%|*}" in
      missing) missing=$((missing + 1));;
      unverified) unverified=$((unverified + 1));;
    esac
  done
  if [ "$missing" -gt 0 ] || [ "$unverified" -gt 0 ]; then
    say "NOT USABLE: $missing of 3 teacher(s) not installed, $unverified present but unverified."
    say "            Soft labels CANNOT be generated here. Use the mock teacher for harness"
    say "            testing only -- it is NOT a physical model (spec §5.7)."
  else
    say "all three teachers present and identity-confirmed; record the versions in the lock file."
  fi
  if [ "$TURNER_PARAMS_VERSION" = "待核验" ]; then
    say "TURNER PARAMETERS UNVERIFIED: set --turner-params-version before generating soft labels."
  fi
  NOT_USABLE=$((missing + unverified))
}

# ------------------------------------------------------------------ commands
print_commands() {
  cat <<EOF
[deploy_teachers] exact install commands for the pinned versions
  (run these on the cluster; this environment has no network access)

  # --- ViennaRNA (provides RNAfold + the McCaskill partition function) ---
  conda install -y -c bioconda viennarna=\${VIENNA_VERSION}          # or:
  pip install ViennaRNA==\${VIENNA_VERSION}
  RNAfold --version                                                  # record the output

  # --- RNAstructure (provides Fold / partition) ---
  conda install -y -c bioconda rnastructure=\${RNASTRUCTURE_VERSION}  # or build from source:
  #   wget https://rna.urmc.rochester.edu/Releases/RNAstructure-X.tar.gz && make
  Fold --version                                                     # record the output

  # --- LinearPartition (linear-time partition function) ---
  conda install -y -c bioconda linearpartition=\${LINEARPARTITION_VERSION}
  linearpartition --version                                          # record the output

  # --- Turner parameters (version-sensitive!) ---
  # ViennaRNA ships rna_turner1999.par / rna_turner2004.par under share/ViennaRNA;
  # the file actually used must be named in the lock file, because the same
  # ViennaRNA version can be built against different parameter sets.

  # --- after installing, re-run ---
  $0 --check --lock-file $LOCK_FILE

EOF
}

# ------------------------------------------------------------------ install
guarded_install() {
  # An unpinned install is not a version lock, so it is refused outright: the
  # whole point of this step is that the soft labels are reproducible.
  if [ "$VIENNA_VERSION" = "待核验" ] || [ "$RNASTRUCTURE_VERSION" = "待核验" ] \
     || [ "$LINEARPARTITION_VERSION" = "待核验" ]; then
    say "REFUSING to install: the pinned versions are still 待核验."
    say "Fill them in (--vienna-version / --rnastructure-version / --linearpartition-version)"
    say "from the vendor release pages, then re-run. Commands are printed instead:"
    echo
    print_commands
    return 1
  fi
  say "checking network reachability before attempting an install"
  local reachable=1
  if command -v getent >/dev/null 2>&1; then
    getent hosts pypi.org >/dev/null 2>&1 || reachable=0
  elif command -v curl >/dev/null 2>&1; then
    curl -sI --max-time 8 https://pypi.org >/dev/null 2>&1 || reachable=0
  else
    reachable=0
  fi
  if [ "$reachable" != "1" ]; then
    say "REFUSING to install: no network reachability to a package index."
    say "This is expected here (spec §3.4). The commands are printed instead:"
    echo
    print_commands
    return 1
  fi
  say "network reachable; running the pinned installs"
  "$PY" -m pip install "ViennaRNA==${VIENNA_VERSION}" || return 1
  conda install -y -c bioconda "rnastructure=${RNASTRUCTURE_VERSION}" || return 1
  conda install -y -c bioconda "linearpartition=${LINEARPARTITION_VERSION}" || return 1
  return 0
}

# ------------------------------------------------------------------ lock file
write_lock() {
  local vienna_status="${VIENNA%%|*}"; local vienna_rest="${VIENNA#*|}"
  local vienna_version="${vienna_rest%%|*}"; local vienna_path="${vienna_rest#*|}"
  local rna_status="${RNASTR%%|*}"; local rna_rest="${RNASTR#*|}"
  local rna_version="${rna_rest%%|*}"; local rna_path="${rna_rest#*|}"
  local lp_status="${LINEARP%%|*}"; local lp_rest="${LINEARP#*|}"
  local lp_version="${lp_rest%%|*}"; local lp_path="${lp_rest#*|}"

  if [ "$DRY" = "1" ]; then
    say "[dry-run] would write the lock file to $LOCK_FILE"
    return 0
  fi
  mkdir -p "$(dirname "$LOCK_FILE")"
  VIENNA_STATUS="$vienna_status" VIENNA_VERSION_DETECTED="$vienna_version" \
  VIENNA_PATH="$vienna_path" RNA_STATUS="$rna_status" RNA_VERSION_DETECTED="$rna_version" \
  RNA_PATH="$rna_path" LP_STATUS="$lp_status" LP_VERSION_DETECTED="$lp_version" \
  LP_PATH="$lp_path" TURNER_FILES="$TURNER_FILES" \
  "$PY" - "$LOCK_FILE" "$VIENNA_VERSION" "$RNASTRUCTURE_VERSION" \
          "$LINEARPARTITION_VERSION" "$TURNER_PARAMS_VERSION" <<'PYEOF'
import json, os, socket, sys, datetime

(lock_path, vienna_pin, rna_pin, lp_pin, turner_pin) = sys.argv[1:6]
UNVERIFIED = "待核验"

def entry(pin, status, detected, path, note):
    return {
        "pinned_version": pin,
        "detected_version": detected,
        "status": status,                       # present | missing
        "path": path or None,
        "pin_verified": pin != UNVERIFIED,
        "note": note,
    }

lock = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "purpose": ("System-2 teacher version lock: the soft labels are only reproducible "
                "if the software AND the Turner parameter set are pinned (spec §3.5, §5.7, §8.1 G5)"),
    "ensemble": "p^teacher = mean(p^ViennaRNA, p^RNAstructure, p^LinearPartition)",
    "teachers": {
        "viennarna": entry(vienna_pin, os.environ["VIENNA_STATUS"],
                           os.environ["VIENNA_VERSION_DETECTED"], os.environ["VIENNA_PATH"],
                           "provides RNAfold and the McCaskill partition function"),
        "rnastructure": entry(rna_pin, os.environ["RNA_STATUS"],
                              os.environ["RNA_VERSION_DETECTED"], os.environ["RNA_PATH"],
                              "ensemble member"),
        "linearpartition": entry(lp_pin, os.environ["LP_STATUS"],
                                 os.environ["LP_VERSION_DETECTED"], os.environ["LP_PATH"],
                                 "linear-time partition function; report its speed-up separately (Q11)"),
    },
    "turner_parameters": {
        "pinned_version": turner_pin,
        "detected_files": os.environ["TURNER_FILES"],
        "pin_verified": turner_pin != UNVERIFIED,
        "note": ("the Turner parameter set changes between releases; the file actually "
                 "used must be named here or the soft labels are not reproducible"),
    },
    "unverified_fields": [k for k, v in (
        ("viennarna.pinned_version", vienna_pin),
        ("rnastructure.pinned_version", rna_pin),
        ("linearpartition.pinned_version", lp_pin),
        ("turner_parameters.pinned_version", turner_pin),
    ) if v == UNVERIFIED],
}
with open(lock_path, "w") as fh:
    json.dump(lock, fh, indent=1, ensure_ascii=False, sort_keys=True)
print(f"[deploy_teachers] wrote {lock_path} "
      f"({len(lock['unverified_fields'])} field(s) still 待核验)")
PYEOF
}

# ------------------------------------------------------------------ main
NOT_USABLE=0
[ "$DO_CHECK" = "1" ] && report
[ "$DO_COMMANDS" = "1" ] && { echo; print_commands; }
if [ "$DO_INSTALL" = "1" ]; then
  guarded_install || NOT_USABLE=$((NOT_USABLE + 1))
fi
if [ "$DO_CHECK" = "1" ]; then
  write_lock
fi

if [ "$NOT_USABLE" -gt 0 ]; then
  say "RESULT: not ready to generate teacher soft labels ($NOT_USABLE teacher(s) missing or unverified / install refused)."
  exit 4
fi
say "RESULT: teacher deployment ready."
exit 0
