# HyperVerITAS
## HyperVerITAS code
Code for HyperVerITAS is contained in **hyperveritas/hello_world/examples**. To run the code, do the following:

1. run the python file **hyperveritas/hello_world/test/helper.py** to generate the image files. It will query you for an input size. If you enter a number *n*, it will generate an image of size 2^n.
2. go to the **hyperveritas/hello_world** directory. run the command **'cargo run --release --example file_name n'**, where file_name is the name of the rust file you wish to run, and n is the input size.

Example: **'cargo run --release --example hv_crop_brakedown 17'** runs HyperVerITAS (Brakedown) full system crop for an image of size 2^17.
