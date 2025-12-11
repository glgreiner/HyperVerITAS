## VerITAS FRI
This directory contains the Full System Implementation for VerITAS's proof system with the FRI Univariate PCS. The original imlpementation only had code for one-channel image transformation proofs and hash pre-image proofs. We took their one-channel code from the original repository, and made 3-channel variants via parallelization (as is suggested in the VerITAS paper) to enable a fair comparison with HyperVerITAS. 

## Installation and Setup

> [!NOTE]
> If you have not already installed the necessary dependencies (Rust, Python, time), install them as described <a href=https://github.com/glgreiner/HyperVerITAS/blob/main/README.md#dependencies>here</a>.

### Creating Sample Images

- Navigate to the directory `HyperVerITAS/comparisons/VerITAS_FRI/images`

- If you already have a python environment active, first deactivate it:

  - `deactivate` 

- Create a new Python environment:

  - `python3 -m venv veritas_fri`
  - `source veritas_fri/bin/activate`
  - `pip install -r requirements.txt`

- Create the sample images by running:

  - `python helper.py`

## Running the Code

Ensure that you are in the `HyperVerITAS/comparisons/VerITAS_FRI` directory.

To run the Full System Implementation for VerITAS FRI, run the following commands:

- Crop: `/usr/bin/time -v cargo run --release --example fullCropFri <size>`

- Grayscale: `/usr/bin/time -v cargo run --release --example fullGrayFri <size>`

Where `<size>` is the input size (2^size number of pixels). Valid choices are numbers between 19-25.
