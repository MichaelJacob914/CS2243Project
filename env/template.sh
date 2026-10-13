# Template for a new machine. Copy to env/<site>.sh, set DATA_ROOT, then:
#   source env/<site>.sh none
#   scripts/setup.sh                                      # builds $DATA_ROOT/venvs/{libero,aloha}
#   python3 scripts/download_ckpts.py pi0_libero pi0_aloha_sim
#   source env/<site>.sh

export CS2243_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT=/path/with/50GB/free                 # EDIT ME

export OPENPI_DATA_HOME=$DATA_ROOT/openpi_data
export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/python
export LIBERO_CONFIG_PATH=$DATA_ROOT/libero_config
export RUNS_ROOT=$DATA_ROOT/runs/$USER
export UV_CACHE_DIR=$DATA_ROOT/cache/$USER/uv
export JAX_COMPILATION_CACHE_DIR=$DATA_ROOT/cache/$USER/jax

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$CS2243_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

_cs2243_venv=${1:-libero}
if [ "$_cs2243_venv" != none ]; then
    source "$DATA_ROOT/venvs/$_cs2243_venv/bin/activate" 2>/dev/null \
        || echo "venv '$_cs2243_venv' not built yet (run scripts/setup.sh)" >&2
fi
unset _cs2243_venv
