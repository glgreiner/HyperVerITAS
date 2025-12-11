pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/bitify.circom";
include "./sha256round.circom";


template Sha256Rounds(num,rounds){
    // Check that the input size is equal to 512 per round
    assert((rounds * 512) == num * 8);


    signal input in[num];
    signal input previous_state[32 * 8];
    signal output next_state[32 * 8];

    // Convert the input to bits
    signal b_in [num * 8];

    component B2b_in[num];
    for (var i = 0; i < num; i++) {
        B2b_in[i] = Num2Bits(8);
        B2b_in[i].in <== in[i];
        for(var j = 0; j < 8; j++)
            b_in[i*8 + j] <== B2b_in[i].out[7-j];
        
    }


    // Compoute the next state
    component sha256round[rounds];
    for(var i = 0;i < rounds; i++){
        sha256round[i] = Sha256Round();
        
        for(var j = 0; j < 512; j++)
            sha256round[i].in[j] <== b_in[i*(512) + j];    

        if(i == 0)
            for(var j = 0; j < 32*8; j++)
                sha256round[i].previous_round[j] <== previous_state[j];
        else
            for(var j = 0; j < 32*8; j++)
                sha256round[i].previous_round[j] <== sha256round[i-1].out[j];
        
    }

    for(var i = 0;i < 32 * 8; i++)
        next_state[i] <== sha256round[rounds-1].out[i];  
}

//MAIN component main = Sha256Rounds(NUM, ROUNDS);