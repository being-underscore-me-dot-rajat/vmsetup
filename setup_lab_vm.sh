#!/usr/bin/env bash
# Sets up a Linux (Ubuntu/Debian) machine or VM for the NSSF Enterprise AI & LLM Engineering course.
# The same steps as setup_lab_vm.ps1: software, VS Code + extensions, the course Python environment with every
# library from Days 1-10, GPU PyTorch if there's an NVIDIA GPU, spaCy model, Jupyter kernel, Ollama and
# Hugging Face model downloads, and a final import check.
#
# Usage (from the VM_Setup folder):   bash setup_lab_vm.sh
# Options (environment variables):   COURSE_ROOT=/opt/nssf   ENV_PATH=$HOME/fund-ai-venv   SKIP_MODELS=1
# Needs sudo for the system packages. Safe to run more than once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COURSE_ROOT="${COURSE_ROOT:-$(dirname "$SCRIPT_DIR")}"
ENV_PATH="${ENV_PATH:-$COURSE_ROOT/DAY1/Labs/work/fund-ai/.venv}"
OLLAMA_MODELS="qwen2.5:1.5b llama3.2:1b gemma2:2b qwen2.5:0.5b"

step() { printf '\n==== %s\n' "$1"; }

step "1. Software"
sudo apt-get update -y
sudo apt-get install -y git curl wget gpg build-essential software-properties-common
# Python 3.13 (the version the course is tested on), from the deadsnakes archive on Ubuntu.
if ! command -v python3.13 >/dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
fi
# Visual Studio Code, from Microsoft's package repository.
if ! command -v code >/dev/null; then
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /tmp/packages.microsoft.gpg
    sudo install -o root -g root -m 644 /tmp/packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
        | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null
    sudo apt-get update -y
    sudo apt-get install -y code
fi
# Ollama.
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

step "2. VS Code extensions"
for ext in ms-python.python ms-python.vscode-pylance ms-toolsai.jupyter; do
    code --install-extension "$ext" --force >/dev/null || echo "  could not install $ext"
done

step "3. Course Python environment: $ENV_PATH"
if [ ! -x "$ENV_PATH/bin/python" ]; then
    mkdir -p "$(dirname "$ENV_PATH")"
    python3.13 -m venv "$ENV_PATH"
fi
PY="$ENV_PATH/bin/python"
"$PY" -m pip install --quiet --upgrade pip
echo "  installing every course library (first time: 10-20 minutes) ..."
"$PY" -m pip install --quiet -r "$SCRIPT_DIR/requirements-all.txt"
if command -v nvidia-smi >/dev/null; then
    # On Linux the default PyTorch wheel already includes CUDA; just report it.
    "$PY" -c "import torch; print('  PyTorch GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available - check the NVIDIA driver')"
else
    echo "  no NVIDIA GPU found: PyTorch will run on the CPU"
fi

step "4. Models, kernel and downloads"
"$PY" -c "import en_core_web_sm" 2>/dev/null || "$PY" -m spacy download en_core_web_sm --quiet
"$PY" -m ipykernel install --user --name fund-ai --display-name "Python (fund-ai)" >/dev/null
if [ "${SKIP_MODELS:-0}" != "1" ]; then
    curl -s localhost:11434/api/version >/dev/null || (nohup ollama serve >/dev/null 2>&1 & sleep 5)
    for m in $OLLAMA_MODELS; do echo "  ollama pull $m"; ollama pull "$m"; done
    "$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "distilbert-base-multilingual-cased",
          "intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]:
    print("   ", m)
    snapshot_download(m)
EOF
fi

step "5. Check"
"$PY" - <<'EOF'
import importlib
mods = ['numpy','pandas','sklearn','xgboost','lightgbm','statsmodels','prophet','dask','pyarrow','torch','transformers',
        'peft','sentence_transformers','faiss','spacy','gensim','imblearn','pyod','fastapi','mlflow','langchain_ollama','dvc']
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
import torch
print("  ", len(mods) - len(bad), "of", len(mods), "libraries import | PyTorch", torch.__version__,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
for b in bad:
    print("  FAILED", b)
EOF
echo
echo "Done. Environment: $PY"
echo "To measure this machine: $PY $SCRIPT_DIR/benchmark_lab_vm.py"
