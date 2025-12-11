# Artifact Appendix (Required for all badges)

Paper title: **HyperVerITAS: Verifying Image Transformations at Scale on Boolean Hypercubes**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [ ] **Reproduced**


## Description (Required for all badges)

1. HyperVerITAS: Verifying Image Transformations at Scale on Boolean Hypercubes
2. This artifact contains an implementation of our proof system, as well as extensions of prior work that we compare to.

### Security/Privacy Issues and Ethical Concerns (Required for all badges)

None

## Basic Requirements (Required for Functional and Reproduced badges)

The Reviewer will only need access to the **Compute VM** in HotCRP to replicate the work.

### Hardware Requirements (Required for Functional and Reproduced badges)

Replace this with the following:

1. It can run on commodity hardware. It needs a little under 100GB of disk space in total. The specs of the laptop will determine how far the proof systems scale, but all proof systems should be able to run on small inputs on a commodity laptop.
2. Not applying for Reproduced

### Software Requirements (Required for Functional and Reproduced badges)

Replace this with the software required to run your artifact and its versions,
as follows.

1. Used Ubuntu 24.04 for the artifcat evaluation. The artifact has also run on MacOS (what we used to get results in the paper), but the instructions included in the github won't work for MacOS.
2. Included in install scripts
3. Didn't use docker, have install scripts.
4. We use Rust and Python in our artifact. Specifically, we use Python 3.12.1 and primarily latest version of Rust.
5. See requirements files in the github repos, as well as explanations in the READMEs
6. None
7. None

### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

- Overall disk space is roughly 70GB
- Overall running time is roughly a few hours (if running all experiments)

## Environment (Required for all badges)

You should use the Compute VM on HotCRP. You can then clone the github repo and follow instructions from there.

### Accessibility (Required for all badges)

Here is the github: https://github.com/glgreiner/HyperVerITAS.git

### Set up the environment (Required for Functional and Reproduced badges)

See the READMEs included in the github repo for how to setup and run the code.

### Testing the Environment (Required for Functional and Reproduced badges)

See the READMEs included in the github repo for how to setup and run the code.

## Artifact Evaluation (Required for Functional and Reproduced badges)

This section should include all the steps required to evaluate your artifact's
functionality and validate your paper's key results and claims. Therefore,
highlight your paper's main results and claims in the first subsection. And
describe the experiments that support your claims in the subsection after that.

### Main Results and Claims

#### Main Result 1: Full System Crop on Laptop

Table 3 in our paper compares HyperVerITAS to other existing image provenance proof systems for proving the crop transformation on a laptop. We recorded four metrics: prover runtime, verifier runtime, prove peak memory, and proof size. We found that HyperVerITAS outperforms prior work (VerITAS, VIMz, and TilesProof) in prover time for cropping 50% of the image on a laptop for various image sizes (2^19 to 2^25).

#### Main Result 2: HyperVerITAS vs VerITAS Crop on Laptop

In Figure 7 (the top part), we compare HyperVerITAS to VerITAS when instantiated with a variety of different Polynomial Commitment Schemes. Notice that regardless of the PCS HyperVerITAS is instantiated with, it outperforms VerITAS in prover runtime (as well as memory).

#### Main Result 3: Full System Grayscale on Laptop

Table 7 in our paper compares HyperVerITAS to other existing image provenance proof systems for proving the grayscale transformation on a laptop. We recorded four metrics: prover runtime, verifier runtime, prove peak memory, and proof size. We found that HyperVerITAS outperforms prior work (VerITAS, VIMz, and TilesProof) in prover time for grayscale on a laptop for various image sizes (2^19 to 2^25).

### Experiments

#### Experiment 1: Full System Crop
- Time: 30 minutes Human time + 3 compute hours
- Storage: ~70GB

You need to run the crop experiments for:
- VerITAS KZG `(comparisons/VerITAS_KZG/README.md)`
- VerITAS FRI `(comparisons/VerITAS_FRI/README.md)`
- HyperVerITAS PST `(./README.md)`
- HyperVerITAS Brakedown (127) `(./README.md)`
- VIMz `(comparisons/vimz/README.md)`
- TilesProof `(comparisons/TilesProof/README.md)`

Details on how to run these experiments are included in the respective README files. For each of the above Schemes, the crop experiment needs to be run for input sizes 2^19, 2^20, ..., 2^25 (or until it crashes due to memory error).

These experiments support Claim 1, as these results were used to make Table 3.

We expect to see HyperVerITAS Brakedown (127) outperform all other schemes in prover runtime.

#### Experiment 2: HyperVerITAS vs VerITAS Crop on Laptop

- Time: 30 minutes Human time + 6 compute hours
- Storage: ~20GB

You need to run the crop experiments for:
- VerITAS KZG `(comparisons/VerITAS_KZG/README.md)`
- VerITAS FRI `(comparisons/VerITAS_FRI/README.md)`
- HyperVerITAS PST `(./README.md)`
- HyperVerITAS Brakedown (64) `(./README.md)`
- HyperVerITAS Brakedown (127) `(./README.md)`
- HyperVerITAS Brakedown (256) `(./README.md)`
- HyperVerITAS Basefold `(./README.md)`
- HyperVerITAS BasefoldFri `(./README.md)`
- HyperVerITAS ZeromorphFri `(./README.md)`

Details on how to run these experiments are included in the respective README files. For each of the above Schemes, the crop experiment needs to be run for input sizes 2^19, 2^20, ..., 2^25 (or until it crashes due to memory error).

These experiments support Claim 2, as these results were used to make the top graph seen in Figure 7.

We expect to see HyperVerITAS (with any PCS) outperform VerITAS in prover runtime.

#### Experiment 3: Full System Grayscale
- Time: 30 minutes Human time + 3 compute hours
- Storage: ~70GB

You need to run the grayscale experiments for:
- VerITAS KZG `(comparisons/VerITAS_KZG/README.md)`
- VerITAS FRI `(comparisons/VerITAS_FRI/README.md)`
- HyperVerITAS PST `(./README.md)`
- HyperVerITAS Brakedown (127) `(./README.md)`
- VIMz `(comparisons/vimz/README.md)`
- TilesProof `(comparisons/TilesProof/README.md)`

Details on how to run these experiments are included in the respective README files. For each of the above Schemes, the grayscale experiment needs to be run for input sizes 2^19, 2^20, ..., 2^25 (or until it crashes due to memory error).

These experiments support Claim 3, as these results were used to make Table 7 in our paper.

We expect to see HyperVerITAS Brakedown (127) outperform all other schemes in prover runtime.

## Limitations (Required for Functional and Reproduced badges)

Although we didn't provide credits to fund an AWS server to replicate the experiments we did on the high-memory AWS server, the main goal of that experiment was to show that even with higher memory, the same trends that we saw on the laptop still hold (i.e HyperVerITAS outperforms VerITAS). So although it won't produce AWS results, the results produced on the laptop should demonstrate the trends effectively. Further, if one had an AWS server, they could clone the github repo and follow the instructions to reproduce those results.

## Notes on Reusability (Encouraged for all badges)

Our artifact not only provides an implementation for HyperVerITAS, but also provides some value beyond our proof scheme. In particular, we found a bug in the implementation of the Brakedown PCS in the plonkish_basefold repository during our implementation that caused errors in the verify method. We solved this bug and provided a solution in our forked version of plonkish_basefold (which one can find in the HyperVerITAS github repo). This is useful to the broader research community, as they can now utilize this implementation to the fullest. 
