This implementation is a fork of the github repository VIMz found <a href="https://github.com/zero-savvy/vimz/tree/pets-2025-artifact">here</a>.

## VIMz setup

1) Ensure you are in the directory: `HyperVerITAS/comparisons/vimz`

2) Run the setup script as follows:
```./vimz_setup.sh```

## Benchmarks

1) Ensure you are still in the directory: `HyperVerITAS/comparisons/vimz`

2) Next, activate the python environment
   
```
source py_modules/vimz/bin/activate
```

3) Run the benchmark script as follows:

```
./benchmark.sh <size> <transformation>
```
 - Valid options for `<size>` are numbers from 19-25. These specify the size of the image. If you input 19, the image is of size 2^19 pixels.

 - Valid optinos for `<transformation>` are `crop` and `grayscale` at the moment, as those are the two functions we support.

 - Some example usage:

    - VIMz Crop for input image size 2^19
      
    ```./benchmark.sh 19 crop```
   
    - VIMz Grayscale for input image size 2^22
      
    ```./benchmark.sh 22 grayscale```
