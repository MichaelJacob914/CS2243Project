#!/bin/bash
#SBATCH --job-name=e14_render
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
# Render the exp14 toy animation and all explainer scenes in parallel (one process per scene), then join the scenes.
# Usage: sbatch scripts/exp14/video/render.sbatch [scene names...]   (default: toy animation + all scenes)
set -uo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd $ROOT
VENV=/n/netscratch/olveczky_lab/Lab/charleszhang/videnv
PY=$VENV/bin/python
MEDIA=/n/netscratch/olveczky_lab/Lab/charleszhang/exp14_video/media
STILLS=/n/netscratch/olveczky_lab/Lab/charleszhang/exp14_video/stills
export PATH=$VENV/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 STILLS_DIR=$STILLS/toy
echo "commit $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
SCENES=${@:-toy S01Question S02Toy S03Measure S04Control S05Transport S06Finding1 S07Finding2 S08Finding3 S09Finding4 S10Recommend}
for s in $SCENES; do
  if [ "$s" = toy ]; then
    $PY scripts/exp14/video/toy_anim_render.py > logs/slurm/render_toy.log 2>&1 &
  else
    $PY -m manim render -qm --disable_caching --media_dir ${MEDIA}_$s scripts/exp14/video/explainer.py $s > logs/slurm/render_$s.log 2>&1 &
  fi
done
wait
for s in $SCENES; do echo "== $s"; tail -2 logs/slurm/render_$s.log; done
$PY scripts/exp14/video/assemble.py
