#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Initialize Nova
echo "Initializing Nova..."
cd nova
cargo build
cargo install --path .

# 2. Install node modules
echo "Installing node modules..."
cd ../circuits
npm install

# 3. Build Circuits
echo "Building Circuits..."
source build_comparison_circuits.sh

# 4. Extract images
echo "Extracting Images..."
cd ../samples/JSON
source unpack.sh

# 5. Setup Python environment
echo "Setting up Python Environment..."
cd ../../py_modules
python3 -m venv vimz
source vimz/bin/activate
pip install -r requirements.txt

echo "Setup complete!"

