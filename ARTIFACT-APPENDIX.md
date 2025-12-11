# Artifact Appendix (Required for all badges)

Paper title: **HyperVerITAS: Verifying Image Transformations at Scale on Boolean Hypercubes**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [ ] **Reproduced**


## Description (Required for all badges)
Replace this with the following:

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
3. Didn't use docker, have install scripts;.
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

List all your paper's results and claims that are supported by your submitted
artifacts.

#### Main Result 1: Name

Describe the results in 1 to 3 sentences. Mention what the independent and
dependent variables are; independent variables are the ones on the x-axes of
your figures, whereas the dependent ones are on the y-axes. By varying the
independent variable (e.g., file size) in a given manner (e.g., linearly), we
expect to see trends in the dependent variable (e.g., runtime, communication
overhead) vary in another manner (e.g., exponentially). Refer to the related
sections, figures, and/or tables in your paper and reference the experiments
that support this result/claim. See example below.

#### Main Result 2: Example Name

Our paper claims that when varying the file size linearly, the runtime also
increases linearly. This claim is reproducible by executing our
[Experiment 2](#experiment-2-example-name). In this experiment, we change the
file size linearly, from 2KB to 24KB, at intervals of 2KB each, and we show that
the runtime also increases linearly, reaching at most 1ms. We report these
results in "Figure 1a" and "Table 3" (Column 3 or Row 2) of our paper.

### Experiments
List each experiment to execute to reproduce your results. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes to execute in human and compute times (approximately).
 - How much space it consumes on disk (approximately) (omit if <10GB).
 - Which claim and results does it support, and how.

#### Experiment 1: Name
- Time: replace with estimate in human-minutes/hours + compute-minutes/hours.
- Storage: replace with estimate for disk space used (omit if <10GB).

Provide a short explanation of the experiment and expected results. Describe
thoroughly the steps to perform the experiment and to collect and organize the
results as expected from your paper (see example below). Use code segments to
simplify the workflow, as follows.

```bash
python3 experiment_1.py
```

#### Experiment 2: Example Name

- Time: 10 human-minutes + 3 compute-hours
- Storage: 20GB

This example experiment reproduces
[Main Result 2: Example Name](#main-result-2-example-name), the following script
will run the simulation automatically with the different parameters specified in
the paper. (You may run the following command from the example Docker image.)

```bash
python3 main.py
```

Results from this example experiment will be aggregated over several iterations
by the script and output directly in raw format along with variances and
standard deviations in the `output-folder/` directory. You will also find there
the plots for "Figure 1a" in `.pdf` format and the table for "Table 3" in `.tex`
format. These can be directly compared to the results reported in the paper, and
should not quantitatively vary by more than 5% from expected results.


## Limitations (Required for Functional and Reproduced badges)

Describe which steps, experiments, results, graphs, tables, etc. are _not
reproducible_ with the provided artifact. Explain why this is not
included/possible and argue why the artifact should _still_ be evaluated for the
respective badges.

## Notes on Reusability (Encouraged for all badges)

First, this section might not apply to your artifacts. Describe how your
artifact can be used beyond your research paper, e.g., as a general framework.
The overall goal of artifact evaluation is not only to reproduce and verify your
research but also to help other researchers to re-use and extend your artifacts.
Discuss how your artifacts can be adapted to other settings, e.g., more input
dimensions, other datasets, and other behavior, through replacing individual
modules and functionality or running more iterations of a specific module.
