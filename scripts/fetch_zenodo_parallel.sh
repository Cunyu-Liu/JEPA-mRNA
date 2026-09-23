#!/bin/bash
# Parallel resumable downloader for Zenodo files.
#
# Why this exists: from this cluster, DNS for zenodo.org is poisoned (resolves to
# "::"), and a single TLS connection to Zenodo is throttled to ~14-20 KB/s.
# Connecting to a hard-coded Zenodo IP (bypassing DNS) works, and aggregate
# throughput scales linearly with the number of concurrent connections
# (measured 2026-09-23: 32 conn -> 460 KB/s, 64 conn -> 950 KB/s). So we fetch
# the file as a set of fixed-size byte ranges with a large worker pool, then
# concatenate and checksum-verify.
#
# Usage:
#   fetch_zenodo_parallel.sh <url> <out_path> <total_size> <md5> [jobs] [chunk_mb]
#
# Resumable: completed chunks are kept in <out_path>.parts/ and skipped on re-run.
set -uo pipefail

URL="$1"; OUT="$2"; SIZE="$3"; MD5="$4"
JOBS="${5:-128}"
CHUNK_MB="${6:-8}"

IPLIST="${ZENODO_IPS:-137.138.52.235 137.138.153.219 188.184.98.114 188.184.103.118 188.185.43.153}"
read -r -a IPS <<< "$IPLIST"
NIP=${#IPS[@]}
CHUNK=$((CHUNK_MB * 1024 * 1024))
PARTS="${OUT}.parts"
NCHUNK=$(( (SIZE + CHUNK - 1) / CHUNK ))

mkdir -p "$PARTS"

log() { echo "[$(date '+%F %T')] $*"; }

log "file=$OUT size=$SIZE chunks=$NCHUNK chunk=${CHUNK_MB}MB jobs=$JOBS"

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$SIZE" ]; then
  got=$(md5sum "$OUT" | cut -d' ' -f1)
  if [ "$got" = "$MD5" ]; then log "ALREADY COMPLETE md5 ok"; exit 0; fi
  log "existing file size ok but md5 mismatch ($got) -> re-downloading"
fi

missing() {
  local n=0 i size
  for ((i=0; i<NCHUNK; i++)); do
    printf -v p "%s/p%06d" "$PARTS" "$i"
    if [ ! -s "$p" ]; then n=$((n+1)); continue; fi
    # expected size of this chunk
    if [ $((i+1)) -eq "$NCHUNK" ]; then exp=$((SIZE - i*CHUNK)); else exp=$CHUNK; fi
    size=$(stat -c%s "$p")
    if [ "$size" != "$exp" ]; then rm -f "$p"; n=$((n+1)); fi
  done
  echo "$n"
}

fetch_one() {
  local i="$1" start end idx ip out
  start=$((i*CHUNK)); end=$((start+CHUNK-1))
  if [ "$end" -ge "$SIZE" ]; then end=$((SIZE-1)); fi
  out=$(printf "%s/p%06d" "$PARTS" "$i")
  idx=$((RANDOM % NIP))
  ip=$(echo "$IPLIST" | cut -d' ' -f$((idx+1)))
  curl -sS --fail --resolve "zenodo.org:443:${ip}" \
       -r "${start}-${end}" --max-time 900 --retry 3 --retry-delay 2 \
       -o "${out}.tmp" "$URL" >/dev/null 2>&1 \
    && mv "${out}.tmp" "$out"
  rm -f "${out}.tmp"
}
export -f fetch_one
export URL SIZE CHUNK PARTS NIP IPLIST

loop=0
while : ; do
  loop=$((loop+1))
  n=$(missing)
  log "pass $loop: missing chunks = $n"
  if [ "$n" -eq 0 ]; then break; fi
  have=$((NCHUNK - n))
  log "progress: $have/$NCHUNK chunks ($(( have * 100 / NCHUNK ))%)"
  seq 0 $((NCHUNK-1)) | xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {}
  n2=$(missing)
  if [ "$n2" -ge "$n" ] && [ "$n2" -gt 0 ]; then
    log "WARN no progress in this pass ($n -> $n2); sleeping 30s before retry"
    sleep 30
  fi
done

log "all chunks present; concatenating"
rm -f "$OUT.tmpcat"
for ((i=0; i<NCHUNK; i++)); do
  printf -v p "%s/p%06d" "$PARTS" "$i"
  cat "$p" >> "$OUT.tmpcat"
done
mv "$OUT.tmpcat" "$OUT"

sz=$(stat -c%s "$OUT")
got=$(md5sum "$OUT" | cut -d' ' -f1)
if [ "$sz" = "$SIZE" ] && [ "$got" = "$MD5" ]; then
  log "SUCCESS size=$sz md5=$got"
  rm -rf "$PARTS"
  exit 0
else
  log "FAILED size=$sz (expect $SIZE) md5=$got (expect $MD5)"
  exit 1
fi