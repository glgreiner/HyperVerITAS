pragma circom 2.0.0;

include "utils/optimized_crop_step.circom";

component main { public [step_in] } = CropHash(204, 102, 2048, 0, 0);