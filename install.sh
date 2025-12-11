#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 0. Init Submodules
git submodule update --init

# 1. Install Rust (nightly)
echo "Installing Rust nightly..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Source Rust environment
. "$HOME/.cargo/env"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
rustup install nightly
rustup default nightly

# 2. Install Python & dependencies
echo "Updating system and installing Python..."
sudo apt update
sudo apt install -y python3-full python3-dev build-essential python3-pip

# 3. Install time
echo "Installing 'time' utility..."
sudo apt install -y time

# 5. Install nodejs + snarkjs
echo "Installing nodejs"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
source ~/.bashrc
nvm install v16.20.0

echo "Installing snarkjs"
npm install -g snarkjs

# 4. Install circom
echo "Installing circom"
cd comparisons
sudo apt install gcc build-essential nlohmann-json3-dev libgmp3-dev nasm
git clone https://github.com/iden3/circom.git
cd circom
rustup default stable
cargo build --release
cargo install --path circom

# 5. Install rapidsnark
echo "Installing rapidsnark"
cd ..
sudo apt-get update
sudo apt-get install build-essential cmake libgmp-dev libsodium-dev nasm curl m4
sudo apt-get install libgl1
git clone https://github.com/iden3/rapidsnark.git
cd rapidsnark
git submodule update -—init
./build_gmp.sh host
make host

echo "Installation complete!"

