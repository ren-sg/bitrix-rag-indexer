# scripts/env_cuda.sh
export VENV_SITE="$PWD/.venv-cuda/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/curand/lib:$VENV_SITE/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
