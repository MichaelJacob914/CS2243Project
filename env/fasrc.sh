# CS2243Project settings for FASRC Cannon. The only file with cluster paths.
#
#   source env/fasrc.sh          # LIBERO venv (default)
#   source env/fasrc.sh aloha    # ALOHA-sim venv
#   source env/fasrc.sh none     # paths only, no venv (for scripts/setup.sh)
#
# For another machine (HRT, laptop), copy env/template.sh to env/<site>.sh and edit it.

export CS2243_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT=/n/netscratch/hankyang_lab/Everyone/mjammot/cs2243

# Shared, read by everyone (built once by scripts/setup.sh + scripts/download_ckpts.py)
export OPENPI_DATA_HOME=$DATA_ROOT/openpi_data       # checkpoints + tokenizer, openpi's cache layout
export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/python       # interpreter behind the shared venvs
export LIBERO_CONFIG_PATH=$DATA_ROOT/libero_config   # pre-written config.yaml, skips LIBERO's first-run prompt

# Per user, written by you
export RUNS_ROOT=$DATA_ROOT/runs/$USER
export UV_CACHE_DIR=$DATA_ROOT/cache/$USER/uv
export JAX_COMPILATION_CACHE_DIR=$DATA_ROOT/cache/$USER/jax

# Headless MuJoCo rendering. MUJOCO_EGL_DEVICE_ID is set per job in slurm/common.sh.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY

# JAX grabs 75 % of GPU memory by default; allocate on demand so rendering fits alongside it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export PATH="$HOME/.local/bin:$PATH"                 # uv, if you build venvs yourself
export PYTHONPATH="$CS2243_REPO/src${PYTHONPATH:+:$PYTHONPATH}"   # your clone's code, not the venv's

_cs2243_venv=${1:-libero}
if [ "$_cs2243_venv" != none ]; then
    if [ -f "$DATA_ROOT/venvs/$_cs2243_venv/bin/activate" ]; then
        source "$DATA_ROOT/venvs/$_cs2243_venv/bin/activate"
    else
        echo "env/fasrc.sh: venv '$_cs2243_venv' not built yet (run scripts/setup.sh)" >&2
    fi
fi
unset _cs2243_venv
