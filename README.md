# HyperVerITAS
This repository contains the implementation for the HyperVerITAS proof system (crop and grayscale transformations). It supports a variety of Multilinear PCS: PST, Brakedown, Basefold, BasefoldFri, and ZeromorphFri. The code for HyperVerITAS is contained in the `hyperveritas_impl` directory. 

We also include code for performing a fair comparison with VerITAS, VIMz, and TilesProof. See the `comparisons` directory for more information.

## General Installation
> [!NOTE]
> This code has run on multiple systems, but we note that it successfully runs on machines with the following specs:
> - Ubuntu @ 24.04
> - rustc @ 1.94.0-nightly
> - python @ 3.12.1
> For Reviewers, these instructions have successfully executed on the `Compute VM` provided in HotCRP.

1) Clone the github repo
```
git clone https://github.com/glgreiner/HyperVerITAS.git
```
2) Run the Installation Script. Note you will need to interact a few times during the installation progress, as you will need to type '1' during the Rust installation, and 'y' a few times during the installation process to accept downloads.
```
cd HyperVerITAS
./install.sh
source ~/.bashrc
```

> [!NOTE]
> This installation script downloads most of the dependencies that all of the systems need to run. However, each proof system (HyperVerITAS, VerITAS, VIMz, TilesProof) is augmented with another setup file, which can be found in their respective directories. You will need to run these as well to finish the setup. Below, we will detail how this setup is done for HyperVerITAS.

## HyperVerITAS Setup

- Run the setup script as follows.
```
./hyperveritas_setup.sh
```

## Benchmarks

  - Navigate to `HyperVerITAS/hyperveritas_impl/`
    
  - You can run HyperVerITAS with the following command:
  ```
  /usr/bin/time -v cargo run --release --example <filename> <size>
  ```
    
  - Valid options for `<filename>` are all of the files in the `hyperveritas_impl/examples` directory that start with `hv`

  - Valid options for `<size>` are numbers from 19-25. These specify the size of the image. If you input 19, the image is of size 2^19 pixels.
      
  - Some example usage:
    
    - HyperVerITAS with Brakedown PCS, Cropping 50%, Image size 2^19
      
      ```/usr/bin/time -v cargo run --release --example hv_crop_brakedown 19```
    - HyperVerITAS with PST PCS, Grayscale, Image size 2^22
      ```/usr/bin/time -v cargo run --release --example hv_gray_pst 22```
  
