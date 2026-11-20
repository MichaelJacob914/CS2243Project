#!/bin/bash
#SBATCH --job-name=auth_wscale
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# Authority (action spread / distance to base) vs steering step k, with w clipped to the actor's
# [-1,1] bound and scaled by c: does a wider bound at late steps recover early-step control?
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
OUT=results/authority_wscale
for t in can square transport; do
  KARG=""; [ $t = transport ] && KARG="--ddim_steps 20"
  python scripts/authority.py --task $t $KARG --out_dir $OUT                       # unclipped c=1, matches old numbers
  for c in 1 1.5 2 3; do python scripts/authority.py --task $t $KARG --clip_w --w_scale $c --out_dir $OUT; done
done
