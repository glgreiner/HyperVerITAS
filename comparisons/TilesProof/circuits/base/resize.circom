pragma circom 2.0.0;
include "../node_modules/circomlib/circuits/comparators.circom";

template Check_Resize(hFull, wFull, hResize, wResize) {

    signal input full_image[hFull][wFull][3];
    signal input resize_image[hResize][wResize][3];

    
    assert((wFull - 1) % (wResize - 1) == 0 && (hFull - 1) % (hResize - 1) == 0);

    var x_ratio = (wFull - 1) / (wResize - 1);
    var y_ratio = (hFull - 1) / (hResize - 1);

    component equals[hResize*wResize*3];
    
    for (var k = 0; k < 3; k++){
        for (var i = 0; i < hResize; i++) {
            for(var j = 0; j < wResize; j++){

                var x_l = ((wFull - 1) * j) \ (wResize - 1) ;
                var y_l = ((hFull - 1) * i) \ (hResize - 1) ;

                var x_h = ((wFull - 1) * j) % (wResize - 1) == 0 ? x_l : x_l + 1;
                var y_h = ((hFull - 1) * i) % (hResize - 1) == 0 ? y_l : y_l + 1;

                var x_weight = (x_ratio * j) - x_l;
                var y_weight = (y_ratio * i) - y_l;

                var a = full_image[y_l][x_l][k];
                var b = full_image[y_l][x_h][k];
                var c = full_image[y_h][x_l][k];
                var d = full_image[y_h][x_h][k];

                var pixel = a * (1 - x_weight) * (1 - y_weight) +
                            b * x_weight * (1 - y_weight) +
                            c * (1 - x_weight) * y_weight +
                            d * x_weight * y_weight;

                equals[i*wResize*3 + j*3 + k] = IsEqual();
                equals[i*wResize*3 + j*3 + k].in[0] <== resize_image[i][j][k];
                equals[i*wResize*3 + j*3 + k].in[1] <== pixel; 
                equals[i*wResize*3 + j*3 + k].out*equals[i*wResize*3 + j*3 + k].out === 1;
                
            }
        }
    }
}

//MAIN component main = Check_Resize(HFULL,WFULL,HRESIZE,WRESIZE);
