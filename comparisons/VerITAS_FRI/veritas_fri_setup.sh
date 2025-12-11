#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Use Rust (nightly)
echo "Use Rust nightly..."
rustup default nightly

# 2. Make output directory
mkdir output

# 3. Initialize Python environment
echo "Setting up Python virtual environment..."
cd images
python3 -m venv veritas_fri
source veritas_fri/bin/activate
pip install -r requirements.txt

# 4. Run helper.py
echo "Running helper.py to generate images..."
python helper.py


echo "All done! VerITAS FRI is setup. Images should be generated."
