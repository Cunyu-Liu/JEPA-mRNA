#!/usr/bin/env bash
set -euo pipefail
export E2B_MODE=shuffled
exec bash /home/rna_jev/tools/run_rinalmo_e2b_round54.sh "$@"
