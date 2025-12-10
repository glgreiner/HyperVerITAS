pragma circom 2.0.0;

include "utils/optimized_crop_step.circom";

component main { public [step_in] } = CropHash(204, 102, 1024, 0, 0);