## VerITAS KZG
This implementation is a fork of the github repository VerITAS found <a href="https://github.com/zk-VerITAS/VerITAS/tree/22e9895e99490bfebfb468cb0425d855a577c742">here</a>.

This directory contains the Full System Implementation for VerITAS's proof system with the KZG Univariate PCS. The original imlpementation only had code for one-channel image transformation proofs and hash pre-image proofs. We took their one-channel code from the original repository, and made 3-channel variants via parallelization (as is suggested in the VerITAS paper) to enable a fair comparison with HyperVerITAS. 

## Installation and Setup

> [!NOTE]
> If you have not already installed the necessary dependencies (Rust, Python, time), install them as described <a href=https://github.com/glgreiner/HyperVerITAS/blob/main/README.md#dependencies>here</a>.

After you have installed the dependencies, we need to change the version of Rust:

- `rustup install nightly-2023-06-13`

- `rustup default nightly-2023-06-13`

### Creating Sample Images

- Navigate to the directory `HyperVerITAS/comparisons/VerITAS_KZG/images`

- If you already have a python environment active, first deactivate it:

  - `deactivate` 

- Create a new Python environment:

  - `python3 -m venv veritas_kzg`
  - `source veritas_kzg/bin/activate`
  - `pip install -r requirements.txt`

- Create the sample images by running:

  - `python helper.py`

## Running the Code

Ensure that you are in the `HyperVerITAS/comparisons/VerITAS_KZG` directory.

Next, create a new directory to hold some output information:

- `mkdir output`

To run the Full System Implementation for VerITAS FRI, run the following commands:

- Crop: `/usr/bin/time -v cargo run --release --example fullCropKZG <size>`

- Grayscale: `/usr/bin/time -v cargo run --release --example fullGrayKZG <size>`

Where `<size>` is the input size (2^size number of pixels). Valid choices are numbers between 19-25.
