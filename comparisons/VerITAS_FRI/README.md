## VerITAS FRI
This directory contains the Full System Implementation for VerITAS's proof system with the FRI Univariate PCS. The original imlpementation only had code for one-channel image transformation proofs and hash pre-image proofs. We took their one-channel code from the original repository, and made 3-channel variants via parallelization (as is suggested in the VerITAS paper) to enable a fair comparison with HyperVerITAS. 

## Running the Code

> [!NOTE]
> If you have not already installed the necessary dependencies (Rust, Python, time), install them as described <a href=https://github.com/glgreiner/HyperVerITAS/blob/main/README.md#dependencies>here</a>.

First, ensure you are in this directory `HyperVerITAS/comparisons/VerITAS_FRI`

To create the images needed to run the code, run the python script:

`cd images`

`python helper.py`

To run the Full System Implementation for VerITAS FRI, you can run the following commands (make sure you are in VerITAS_FRI directory):

`cargo run --release --example fullCropFri n`

`cargo run --release --example fullGrayFri n`

Where `n` is the input size.
