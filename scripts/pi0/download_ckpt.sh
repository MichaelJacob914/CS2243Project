#!/bin/bash
# pi0 LIBERO checkpoint. The s3://openpi-assets bucket is gone; upstream openpi serves it from GCS. The local cache
# path is <OPENPI_DATA_HOME>/openpi-assets/checkpoints/pi0_libero for either scheme, so s3:// loads hit the cache.
#   sbatch -p test -A olveczky_lab -c 4 --mem=16G -t 2:00:00 -o logs/exp7/dl_%j.out scripts/pi0/download_ckpt.sh
set -euxo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export OPENPI_DATA_HOME=/n/netscratch/olveczky_lab/Lab/charleszhang/openpi_cache
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi0_libero', token='anon'))"
du -sh $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_libero
