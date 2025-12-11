pragma circom 2.0.0;


include "./committed_rounds_sha256.circom";
include "./resize.circom";

template Resize_Sha256(num,rounds,h,w,ch,cw){

    signal input  previous_state[32 * 8]; //bit
    signal input next_state[32 * 8]; //bit
    signal input  in[num]; //byte

    signal input image[h][w][3];
    signal input resize_image[ch][cw][3];

    signal output previous_hash; //ge
    signal output next_hash; //ge

    //**************************************************************************
    component sha256_rounds = CommittedRoundSha256(num, rounds);

    for(var i = 0; i < 32 * 8; i++){
        sha256_rounds.previous_state[i] <== previous_state[i];
        sha256_rounds.next_state[i] <== next_state[i];
    }
    for(var i = 0; i < num; i++){
        sha256_rounds.in[i] <== in[i];
    }

    //**************************************************************************
    component resize = Resize(h,w,ch,cw);

    for(var i = 0; i < h; i++)
        for(var j = 0; j < w; j++)
            for(var k = 0; k < 3; k++)
                resize.full_image[i][j][k] <== image[i][j][k];

    for(var i = 0; i < ch; i++)
        for(var j = 0; j < cw; j++)
            for(var k = 0; k < 3; k++)
                resize.resize_image[i][j][k] <== resize_image[i][j][k];

}

//MAIN component main = Resize_Sha256(NUM, ROUNDS, FH, FW, CH, CW);