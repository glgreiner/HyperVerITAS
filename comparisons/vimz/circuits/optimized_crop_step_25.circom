pragma circom 2.0.0;

include "utils/optimized_crop_step.circom";

component main { public [step_in] } = CropHash(819, 409, 4096, 0, 0);