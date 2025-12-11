# HyperVerITAS
Implementation for the HyperVerITAS proof system

## Installation

### Dependencies
  - **a) Rust**:
    - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    - `. "$HOME/.cargo/env"`
    - `rustup install nightly`
    - `rustup default nightly`
  - **b) Python:**
    - `sudo apt update`
    - `sudo apt install python3-full python3-dev build-essential python3-pip`
  - **c) Time**:
    - `sudo apt install time`

> [!NOTE]
> This code has run on multiple systems, but we note that it successfully runs on machines with the following specs:
> - Ubuntu @ 24.04
> - rustc @ 1.94.0-nightly
> - python @ 3.12.1

### Installing HyperVerITAS
  - Clone and Initialize the Github
    - `git clone https://github.com/glgreiner/HyperVerITAS.git`
    - `cd HyperVerITAS`
    - `git submodule update --init`
  - Initialize a python environment
    -  `cd hyperveritas_impl`
    -  `python3 -m venv hyperveritas`
    -  `source hyperveritas/bin/activate`
    -  `cd images`
    -  `pip install -r requirements.txt`
    
#### Creating Sample Images

  - Ensure that you are in `hyperveritas_impl/images`
  - Ensure that you have the `hyperveritas` python environment activated
  - Run the command: `python helper.py`
    - This will create images of size 2^17, 2^18, ..., 2^25

## Benchmarks

  - To run the code, navigate to `HyperVerITAS/hyperveritas_impl/`
  - Run the commmand: `time -v cargo run --release --example <filename> <size>`. Some example usage:
    - HyperVerITAS with Brakedown PCS, Cropping 50%, Image size 2^19
      - `time -v cargo run --release --example hv_crop_brakedown 19`
    - HyperVerITAS with PST PCS, Grayscale, Image size 2^22
      - `time -v cargo run --release --example hv_gray_pst 22`
  
