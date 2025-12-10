Install the appropriate rust nightly version:

`rustup install nightly-2023-06-13`

`rustup default nightly-2023-06-13`

To create the images needed to run the code, run the python script:

`cd images`

`python helper.py`

To run the Full System Implementation for VerITAS KZG, you can run the following commands (make sure you are in VerITAS_KZG directory):

`cargo run --release --example fullCropKZG n`

`cargo run --release --example fullGrayKZG n`

Where `n` is the input size.