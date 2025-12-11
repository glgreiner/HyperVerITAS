#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Install Rust (nightly)
echo "Installing Rust nightly..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Source Rust environment
. "$HOME/.cargo/env"
rustup install nightly
rustup default nightly

# 2. Install Python & dependencies
echo "Updating system and installing Python..."
sudo apt update
sudo apt install -y python3-full python3-dev build-essential python3-pip

# 3. Install time
echo "Installing 'time' utility..."
sudo apt install -y time

# 4. Initialize Python environment
echo "Setting up Python virtual environment..."
cd hyperveritas_impl
python3 -m venv hyperveritas
source hyperveritas/bin/activate

# 5. Install Python dependencies
cd images
pip install -r requirements.txt

# 6. Run helper.py
echo "Running helper.py to generate images..."
python helper.py

echo "All done! Images should be generated."
