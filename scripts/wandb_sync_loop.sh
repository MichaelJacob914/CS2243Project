#!/bin/bash
# Uploads every offline wandb run under logs/adsrl to the `adsrl` project every $INTERVAL seconds.
# Compute nodes have no internet, so runs log offline; run this on a login node:
#   nohup scripts/wandb_sync_loop.sh > logs/wandb_sync.log 2>&1 &
# Stop: kill the PID in logs/wandb_sync.pid (on the same login node, see logs/wandb_sync.host).
# Success = wandb printed "Finished syncing". Its exit code is unreliable: the client sometimes fails
# to delete its temp dir on the NFS scratch after a complete upload.
# --include-synced re-uploads runs that are still growing; finished runs are skipped once their
# last sync postdates their last write.
INTERVAL=${INTERVAL:-900}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
# The project venv has wandb 0.17.7, which fails the current server's settings check and still prints
# "done". So uploads use a current standalone client (uv tool install --python 3.11 wandb).
WANDB_BIN=${WANDB_BIN:-/n/netscratch/olveczky_lab/Lab/charleszhang/wandb_cli/wandb/bin/wandb}
ENTITY=${ENTITY:-charles-zhang216}; PROJECT=${PROJECT:-adsrl}
export TMPDIR=/n/netscratch/olveczky_lab/Lab/charleszhang/tmp; mkdir -p "$TMPDIR"
echo $$ > logs/wandb_sync.pid; hostname > logs/wandb_sync.host
while true; do
  t0=$(date +%s); n=0
  for d in $(ls -d logs/adsrl/*/*/wandb/offline-run-* logs/adsrl/*/*/*/wandb/offline-run-* 2>/dev/null); do
    f=$(ls "$d"/*.wandb 2>/dev/null | head -1); [ -n "$f" ] || continue
    stamp="$d/.last_sync"
    if [ -f "$stamp" ] && [ "$stamp" -nt "$f" ]; then continue; fi  # unchanged since last upload
    touch "$stamp.tmp"
    if timeout 600 "$WANDB_BIN" sync --include-synced -e "$ENTITY" -p "$PROJECT" "$d" > "$TMPDIR/wandb_sync_last.out" 2>&1; grep -q "Finished syncing" "$TMPDIR/wandb_sync_last.out"; then mv "$stamp.tmp" "$stamp"; n=$((n+1));
    else echo "[$(date +%F' '%T)] sync failed: $d"; rm -f "$stamp.tmp"; fi
  done
  echo "[$(date +%F' '%T)] synced $n runs in $(( $(date +%s) - t0 ))s"
  sleep "$INTERVAL"
done
