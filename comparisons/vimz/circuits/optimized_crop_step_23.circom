pragma circom 2.0.0;

include "utils/optimized_crop_step.circom";

component main { public [step_in] } = CropHash(409, 204, 2048, 0, 0);