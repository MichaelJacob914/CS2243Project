#!/bin/bash
# CPU tests of the noise-pattern diagnostic + dry run of its argument parsing.
#   sbatch scripts/pi0/noise_pattern_tests.sbatch
#SBATCH --job-name=p7_noisepat_test
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -q -s -p no:cacheprovider tests/test_noise_pattern.py
for t in 8 4 6; do python scripts/pi0/noise_pattern_eval.py --task $t --dry_run 1; done
python scripts/pi0/noise_pattern_eval.py --task 8 --conds agent_k3 tiled_k8 --episodes 50 --dry_run 1
python scripts/pi0/noise_pattern_eval.py --task 8 --conds bogus_k3 --dry_run 1 && echo "UNEXPECTED: bad cond accepted" || echo "bad cond rejected (expected)"
