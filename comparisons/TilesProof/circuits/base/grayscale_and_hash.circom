pragma circom 2.0.0;

include "./grayscale.circom";
include "./image_hash.circom";

template Grayscale_Hash(height, width){
    signal input image[height][width][3];
    signal input gs_image[height][width];

    signal output out;

    component gs_checker = Grayscale(height, width);
    component hash = ImageHash(height, width);

    //** Grayscale Checker **//

    for (var i = 0; i < height; i++)
        for (var j = 0; j < width; j++) 
            for (var k = 0; k < 3; k++) 
                gs_checker.image[i][j][k] <== image[i][j][k];


    for (var i = 0; i < height; i++)
        for (var j = 0; j < width; j++) 
                gs_checker.gs_image[i][j] <== gs_image[i][j];

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
//MAIN component main{public [gs_image]} = Grayscale_Hash(HEIGHT,WIDTH);
