pragma circom 2.0.3;

include "./resize.circom";
include "./image_hash.circom";

template Resize_Hash(hFull, wFull, hResize, wResize){

    signal input full_image[hFull][wFull][3];
    signal input resize_image[hResize][wResize][3];

    signal output out;

    component resize_checker = Check_Resize(hFull, wFull, hResize, wResize);
    component hash = ImageHash(hFull, wFull);

    //** Resize Checker **//

    for (var i = 0; i < hFull; i++)
        for (var j = 0; j < wFull; j++) 
            for (var k = 0; k < 3; k++) 
                resize_checker.full_image[i][j][k] <== full_image[i][j][k];


    for (var i = 0; i < hResize; i++)
        for (var j = 0; j < wResize; j++) 
            for (var k = 0; k < 3; k++) 
                resize_checker.resize_image[i][j][k] <== resize_image[i][j][k];

    //** Poseidon Hash **// 
    
    for (var i = 0; i < hFull; i++) {
        for (var j = 0; j < wFull; j++) {
            for (var k = 0; k < 3; k++) {
                hash.in[i][j][k] <== full_image[i][j][k];
            }
        }
    }       

    out <== hash.hash;
}
//MAIN component main{public [resize_image]} = Resize_Hash(HFULL,WFULL,HRESIZE,WRESIZE);
