
pragma circom 2.0.0;

include "./sha256rounds.circom";
include "../node_modules/circomlib/circuits/bitify.circom";
include "./bytehash.circom";

template CommittedRoundSha256(num,rounds){

    signal input  previous_state[32 * 8]; //bit
    signal input next_state[32 * 8]; //bit
    signal input  in[num]; //byte

    signal output in_hash; //ge
    signal output previous_hash; //ge
    signal output next_hash; //ge


    //######### COMPUTE SHA256 ROUNDS #########
    component sha256rounds = Sha256Rounds(num, rounds);
    for (var i = 0; i < num; i++)
        sha256rounds.in[i] <== in[i];
    for (var i = 0; i < 32 * 8; i++)
        sha256rounds.previous_state[i] <== previous_state[i];

        //######### CHECK IF THE NEXT STATE IS CORRECT #########

    component equals[32*8];
    for (var i = 0; i < (32*8); i++){
        equals[i] = IsEqual();
        equals[i].in[0] <== sha256rounds.next_state[i];
        equals[i].in[1] <== next_state[i];

        equals[i].out * equals[i].out === 1;
    }

    //######### COMMITMENTS OF THE STATE #########

        //######### COMMITMENT OF THE PREVIOUS STATE #########
    component b2B_previous[32];
    for (var i = 0; i < 32; i++){
        b2B_previous[i] = Bits2Num(8);
        for(var j = 0; j < 8; j++)
            b2B_previous[i].in[j] <== previous_state[i*8 + j];
    }

    component bytehash_previous = ByteHash(32);
    for (var i = 0; i < 32; i++)
        bytehash_previous.in[i] <== b2B_previous[i].out;
    previous_hash <== bytehash_previous.out;

        //######### COMMITMENT OF THE NEXT STATE #########
    component b2B_next[32];
    for (var i = 0; i < 32; i++){
        b2B_next[i] = Bits2Num(8);
        for(var j = 0; j < 8; j++)
            b2B_next[i].in[j] <== next_state[i*8 + j];
    }

    component bytehash_next = ByteHash(32);
    for (var i = 0; i < 32; i++)
        bytehash_next.in[i] <== b2B_next[i].out;
    next_hash <== bytehash_next.out;


}

//MAIN component main = CommittedRoundSha256(NUM, ROUNDS);