#!/usr/bin/env bash
set -e

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "Creating virtual environment in $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install -U pip wheel setuptools

echo "Installing PyTorch..."
# Install from PyTorch index - adjust for your CUDA version
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
pip install -U torch torchvision

echo "Installing dependencies..."
pip install -U "transformers[torch]" peft huggingface-hub datasets accelerate bitsandbytes safetensors sentencepiece protobuf

echo "Installing Heretic..."
pip install -U "git+https://github.com/p-e-w/heretic.git"

echo "Done! Activate with: source $VENV_DIR/bin/activate"
