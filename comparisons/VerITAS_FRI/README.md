## VerITAS FRI
This implementation is a fork of the github repository VerITAS found <a href="https://github.com/zk-VerITAS/VerITAS.git">here</a>.

This directory contains the Full System Implementation for VerITAS's proof system with the FRI Univariate PCS. The original imlpementation only had code for one-channel image transformation proofs and hash pre-image proofs. We took their one-channel code from the original repository, and made 3-channel variants via parallelization (as is suggested in the VerITAS paper) to enable a fair comparison with HyperVerITAS. 

## VerITAS FRI Setup

1) Ensure you are in the directory: `HyperVerITAS/comparisons/VerITAS_FRI`

2) Run the setup script as follows:
```./veritas_fri_setup.sh```

## Benchmarks

1) Ensure that you are in the `HyperVerITAS/comparisons/VerITAS_FRI` directory.

2) To run the Full System Implementation for VerITAS FRI, run the following commands:

- Crop: `/usr/bin/time -v cargo run --release --example fullCropFri <size>`

- Grayscale: `/usr/bin/time -v cargo run --release --example fullGrayFri <size>`

Where `<size>` is the input size (2^size number of pixels). Valid choices are numbers between 19-25.

- The command will print out the **Prover Runtime**, **Verifier Runtime**, **Proof Size**, and **Prover Peak Memory**, the four metrics we record in our paper. The prover peak memory is output in the line titled `Maximum resident set size (kbytes)`.
