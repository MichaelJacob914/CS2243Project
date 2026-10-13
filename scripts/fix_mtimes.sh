#!/bin/bash
# One-time timestamp fix for data newly placed on netscratch.
#
# Netscratch purges files older than 90 days, judged by the file's date. Downloads, conda/uv
# installs and `tar`/`cp -p`/`rsync -a` can bring files dated months or years in the past,
# which the next purge would delete. FASRC allows setting dates once when data is first
# placed; re-touching later to dodge the purge is prohibited.
#
# Run this ONCE as the last step of a download/install/copy. Never on a schedule.
#
# Usage: scripts/fix_mtimes.sh <dir> [<dir> ...]
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <dir> [<dir> ...]" >&2; exit 2; }
today=$(date +%F)
for d in "$@"; do
    [ -e "$d" ] || { echo "$d: does not exist" >&2; exit 1; }
    n=$(find "$d" ! -newermt "$today" | wc -l)
    find "$d" ! -newermt "$today" -exec touch -h {} +
    left=$(find "$d" ! -newermt "$today" | wc -l)
    echo "$d: fixed $n old-dated entries; still old: $left"
done
