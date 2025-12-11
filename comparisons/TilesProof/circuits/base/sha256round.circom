pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/sha256/sha256compression.circom";

template Sha256Round(){


    component sha256compression = Sha256compression();


    signal input previous_round[32*8]; // s[i-1] previous round (256 bit as always)
    signal input in[512]; // 512 bit block
    signal output out[32*8]; // s[i] next round (256 bit as always)





    for (var k=0; k<32; k++ ) {
        sha256compression.hin[32*0+k] <== previous_round[32*0+31-k];
        sha256compression.hin[32*1+k] <== previous_round[32*1+31-k];
        sha256compression.hin[32*2+k] <== previous_round[32*2+31-k];
        sha256compression.hin[32*3+k] <== previous_round[32*3+31-k];
        sha256compression.hin[32*4+k] <== previous_round[32*4+31-k];
        sha256compression.hin[32*5+k] <== previous_round[32*5+31-k];
        sha256compression.hin[32*6+k] <== previous_round[32*6+31-k];
        sha256compression.hin[32*7+k] <== previous_round[32*7+31-k];
    }

    for (var k=0; k<512; k++ ) {
        sha256compression.inp[k] <== in[k];
    }

    for (var k=0; k<32; k++ ) {
        out[32*0+k] <== sha256compression.out[32*0+k];
        out[32*1+k] <== sha256compression.out[32*1+k];
        out[32*2+k] <== sha256compression.out[32*2+k];
        out[32*3+k] <== sha256compression.out[32*3+k];
        out[32*4+k] <== sha256compression.out[32*4+k];
        out[32*5+k] <== sha256compression.out[32*5+k];
        out[32*6+k] <== sha256compression.out[32*6+k];
        out[32*7+k] <== sha256compression.out[32*7+k];
    }

}