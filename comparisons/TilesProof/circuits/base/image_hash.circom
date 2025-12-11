pragma circom 2.0.0;
include "./poseidon_sponge.circom";

template ByteHash(num){
    signal input in[num];
    signal output out;


    component poseidon = SpongeHash(num\28);
    var byte_chuck = 0;
    var hash_counter = 0;

    for(var i = 0; i < num; i++){
        byte_chuck = byte_chuck*(2**8) + in[i];

        if((i+1) % 28 == 0){
            poseidon.in[hash_counter] <== byte_chuck;
            hash_counter++;
            byte_chuck = 0;
        }
    }
    out <== poseidon.hash;
}


template ImageHash(height,width){
    signal input in[height][width][3];
    signal output hash;

    var byte_count = height * width * 3;
    var padding = (byte_count % 28) ? 28 - (byte_count % 28) : 0;
    component byte_hash = ByteHash(byte_count + padding);

    for(var i = 0; i < height; i++)
        for(var j = 0; j < width; j++)
            for(var k = 0; k < 3; k++)
                byte_hash.in[(i*width*3) + (j*3) + k] <== in[i][j][k];

    for(var i = byte_count; i < byte_count + padding; i++){
        if(i == byte_count)
            byte_hash.in[i] <== 7;
        else
            byte_hash.in[i] <== 0;
    }

    hash <== byte_hash.out;
}

//MAIN component main = ImageHash(H,W);
