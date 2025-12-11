This implementation is a fork of the github repository Privacy-PreservingProofs4EditedPhotos found <a href="https://github.com/pierpaolodm/Privacy-PreservingProofs4EditedPhotos">here</a>.

## TilesProof setup

1) Ensure you are in the directory: `HyperVerITAS/comparisons/TilesProof`

2) Run the setup script as follows:
   
```
./tiles_setup.sh
```

## Benchmarks

1) Ensure you are still in the directory: `HyperVerITAS/comparisons/TilesProof`

2) Next, activate the python environment
   
  ```
  source tiles/bin/activate
  ```

3) Run the benchmark script as follows:

    - Crop for tile size 184756
      
      ```
      python generate_proof_crop.py --image ./test/tile_184756.png --N 1 --height 286 --width 323 --height_start 0 --width_start 0 --pot pot25.ptau
      python verify_proof.py --circuit tile_0
      ```
   
    - Grayscale for tile size 80000
      
      ```
      python generate_proof_gray.py --image ./test/tile_80000.png --N 1 --pot pot25.ptau
      python verify_proof.py --circuit tile_0
      ```

 - The command will print out the **Prover Runtime**, **Verifier Runtime**, and **Prover Peak Memory**. To get **Proof Size**, you can manually check the size of the `proof.json` file found in the directory `TilesProof/output/snarkjs_circuit/tile_0`. You can use the command `ls -lh` to see the size of the file in bytes.
   
 - Once you have those metrics for 1 tile, we can generate metrics for any image size. We determine how many tiles are needed to cover the given image, and then multiply each metric (for 1 tile) by that number to obtain the final values.
