# Sourced by slurm/*.sbatch from the repo root: shared env, rendering backend, diagnostic header.
#   source slurm/common.sh [libero|aloha]
# Also works on a login node (no GPU): it falls back to CPU rendering.

source env/fasrc.sh "${1:-libero}"
mkdir -p "$RUNS_ROOT/slurm"
echo "== $(date '+%F %T') $(hostname) job=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader 2>/dev/null || true

# MuJoCo offscreen rendering (EGL). Try NVIDIA's EGL on the job's GPU first. MIG slices (gpu_test) may
# have no graphics support, and FASRC has no system software renderer (mesa-dri-drivers is not
# installed), so the fallback is Mesa's llvmpipe from the conda prefix built by `scripts/setup.sh mesa`.
_probe() {
    python - <<'PY' >/dev/null 2>&1
import mujoco
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
PY
}
export MUJOCO_EGL_DEVICE_ID=0   # index into the EGL devices this job can see
if _probe; then
    echo "== render: NVIDIA EGL (GPU)"
elif [ -d "$DATA_ROOT/mesa/lib/dri" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=$DATA_ROOT/mesa/share/glvnd/egl_vendor.d/50_mesa.json
    export LIBGL_DRIVERS_PATH=$DATA_ROOT/mesa/lib/dri
    export LD_LIBRARY_PATH=$DATA_ROOT/mesa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # llvmpipe render threads
    if _probe; then
        echo "== render: Mesa llvmpipe (CPU, $LP_NUM_THREADS threads) -- no GPU graphics on this device"
    else
        echo "== render: NO working EGL backend -- rendering will fail" >&2
    fi
else
    echo "== render: NVIDIA EGL failed and $DATA_ROOT/mesa is missing (run scripts/setup.sh mesa)" >&2
fi
unset -f _probe
