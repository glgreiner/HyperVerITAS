#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Get Power of Tau file
echo "Downloading Power of Tau file..."
wget https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_25.ptau -O pot25.ptau

# 2. Install Python & dependencies
echo "Setup Python Environment..."
python3 -m venv tiles
source tiles/bin/activate
pip install -r requirements.txt

# 3. Install node modules
echo "Installing node modules"
npm install

echo "All done! TilesProof is now setup"
