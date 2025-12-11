pragma circom 2.0.0;

include "./crop.circom";
include "./image_hash.circom";

template Crop_Hash(height, width, cropped_height, cropped_width, cropped_height_start, cropped_width_start){

    signal input image[height][width][3];
    signal input cropped_image[cropped_height][cropped_width][3];

    signal output out;

    component crop_checker = Crop(height, width, cropped_height, cropped_width, cropped_height_start, cropped_width_start);
    component hash = ImageHash(height, width);

    //** Crop Checker **//

    for (var i = 0; i < height; i++)
        for (var j = 0; j < width; j++) 
            for (var k = 0; k < 3; k++) 
                crop_checker.image[i][j][k] <== image[i][j][k];


    for (var i = 0; i < cropped_height; i++)
        for (var j = 0; j < cropped_width; j++) 
            for (var k = 0; k < 3; k++) 
                crop_checker.cropped_image[i][j][k] <== cropped_image[i][j][k];

    //** Poseidon Hash **// 
    
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            for (var k = 0; k < 3; k++) {
                hash.in[i][j][k] <== image[i][j][k];
            }
        }
    }       

    out <== hash.hash;
}
//MAIN component main{public [cropped_image]} = Crop_Hash(HEIGHT,WIDTH,CH,CW,CSH,CSW);
