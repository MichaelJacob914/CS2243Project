# Source from the project root: environment for .venv-furniture (exp_6, Furniture-Bench/IsaacGym).
source .venv-furniture/bin/activate
# IsaacGym's gym_38.so links libpython3.8.so.1.0 (uv-managed CPython 3.8)
export LD_LIBRARY_PATH=$(python -c "import sys; print(sys.base_prefix)")/lib:${LD_LIBRARY_PATH:-}
# isaacgym.gymtorch is JIT-compiled with torch.utils.cpp_extension; torch 2.4 needs GCC >= 9 (system gcc is 8.5)
GCC_HOME=/n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01
MPFR_LIB=/n/sw/helmod-rocky8/apps/Core/mpfr/4.1.0-fasrc01/lib64  # cc1plus needs libmpfr.so.6 (not on the nodes)
export PATH=$GCC_HOME/bin:$PATH LD_LIBRARY_PATH=$GCC_HOME/lib64:$GCC_HOME/lib:$MPFR_LIB:$LD_LIBRARY_PATH CC=gcc CXX=g++
export TORCH_EXTENSIONS_DIR=$PWD/third_party/torch_extensions
