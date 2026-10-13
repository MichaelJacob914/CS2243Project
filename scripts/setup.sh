#!/bin/bash
# Build the venvs in $DATA_ROOT/venvs/{libero,aloha}.
#
# Both start from openpi's own lockfile (python 3.11, jax 0.5.3, torch 2.7.1), so we inherit its
# tested pins. They differ only in the simulator:
#   aloha  : openpi as locked (gym-aloha 0.1.1 -> mujoco 2.3.7, dm-control 1.0.14)
#   libero : + LIBERO (nakamotoo fork, editable) + robosuite 1.4.1 + mujoco 3.2.3
#            (mujoco 3.2.3 = what openpi's official LIBERO client uses; gym-aloha is unused here)
#
# Nothing in a venv may point into anyone's home (homes are private): openpi is installed
# non-editable, LIBERO is installed editable from a pinned copy in $DATA_ROOT/src, and the
# interpreter lives in $UV_PYTHON_INSTALL_DIR.
#
#   mesa   : Mesa's software renderer (llvmpipe) in $DATA_ROOT/mesa, from conda-forge. CPU rendering
#            fallback for GPUs without graphics support (A100 MIG slices on gpu_test); FASRC has no
#            system software renderer. slurm/common.sh picks it up automatically.
#
# Usage:  source env/fasrc.sh none && scripts/setup.sh [libero] [aloha] [mesa]
set -euo pipefail
: "${DATA_ROOT:?source env/<site>.sh none first}"
: "${CS2243_REPO:?source env/<site>.sh none first}"
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(libero aloha mesa)
export UV_PYTHON_PREFERENCE=only-managed   # interpreter from $UV_PYTHON_INSTALL_DIR, never a home-dir python
mkdir -p "$DATA_ROOT"/{venvs,src,cache/$USER/uv,cache/$USER/jax,runs/$USER}

# Pinned LIBERO copy for the editable install. `tar -m` gives files today's date, not the commit's.
lib_src=$DATA_ROOT/src/LIBERO
lib_commit=$(git -C "$CS2243_REPO/third_party/LIBERO" rev-parse HEAD)
if [ "$(cat "$lib_src/.commit" 2>/dev/null)" != "$lib_commit" ]; then
    echo "== LIBERO @ $lib_commit -> $lib_src"
    rm -rf "$lib_src" && mkdir -p "$lib_src"
    git -C "$CS2243_REPO/third_party/LIBERO" archive HEAD | tar -x -m -C "$lib_src"
    echo "$lib_commit" > "$lib_src/.commit"
fi

# LIBERO config: point it at the pinned copy so it never prompts on first import.
mkdir -p "$DATA_ROOT/libero_config"
b=$lib_src/libero/libero
cat > "$DATA_ROOT/libero_config/config.yaml" <<EOF
benchmark_root: $b
bddl_files: $b/bddl_files
init_states: $b/init_files
datasets: $lib_src/libero/datasets
assets: $b/assets
EOF

for v in "${targets[@]}"; do
    if [ "$v" = mesa ]; then
        conda_bin=$(command -v mamba || command -v conda || true)
        if [ -z "$conda_bin" ]; then
            echo "== [mesa] skipped: needs mamba or conda" >&2
        elif [ ! -d "$DATA_ROOT/mesa/lib/dri" ]; then
            echo "== [mesa] conda-forge mesalib -> $DATA_ROOT/mesa"
            CONDA_PKGS_DIRS=$DATA_ROOT/cache/$USER/conda_pkgs \
                "$conda_bin" create -y -q -p "$DATA_ROOT/mesa" -c conda-forge --override-channels mesalib
        fi
        continue
    fi
    venv=$DATA_ROOT/venvs/$v
    echo "== [$v] openpi lockfile -> $venv"
    # rerun-sdk (lerobot's visualizer, unused by openpi inference) only has glibc >= 2.31 wheels;
    # FASRC's RHEL8 is glibc 2.28, so it is skipped.
    (cd "$CS2243_REPO/third_party/openpi" &&
        GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen --no-editable --python 3.11 \
            --no-install-package rerun-sdk)
    uv pip install --python "$venv/bin/python" pyyaml imageio-ffmpeg
    case $v in
        libero)
            uv pip install --python "$venv/bin/python" \
                mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1 easydict future gym==0.26.2 h5py termcolor matplotlib \
                -e "$lib_src"
            # LIBERO's top-level `libero/` has no __init__.py, so the editable finder misses it
            # (openpi's own instructions add it to PYTHONPATH). A plain .pth does the same per venv.
            site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
            echo "$lib_src" > "$site/libero_src.pth" ;;
        aloha) ;;
        *) echo "unknown venv '$v' (libero|aloha)" >&2; exit 1 ;;
    esac
done

# One-time date fix for everything just placed on netscratch (see scripts/fix_mtimes.sh).
"$CS2243_REPO/scripts/fix_mtimes.sh" "$DATA_ROOT/venvs" "$DATA_ROOT/python" "$DATA_ROOT/src" \
    $( [ -d "$DATA_ROOT/mesa" ] && echo "$DATA_ROOT/mesa" )
echo "== done: ${targets[*]}"
