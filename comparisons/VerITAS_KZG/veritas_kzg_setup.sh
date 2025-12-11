#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Use Rust (nightly)
echo "Use Rust nightly..."
rustup install nightly-2023-06-13
rustup default nightly-2023-06-13

# 2. Make output directory
mkdir output

# 3. Initialize Python environment
echo "Setting up Python virtual environment..."
cd images
python3 -m venv veritas_kzg
source veritas_kzg/bin/activate
pip install -r requirements.txt

# 4. Run helper.py
echo "Running helper.py to generate images..."
python helper.py


echo "All done! VerITAS KZG is setup. Images should be generated."
