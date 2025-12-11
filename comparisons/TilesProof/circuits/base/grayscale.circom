
pragma circom 2.0.0;
include "../node_modules/circomlib/circuits/comparators.circom";

template Grayscale(height, width) {
    signal input image[height][width][3];
    signal input gs_image[height][width];

    component lessthens[height*width];

    for (var i = 0; i <  height; i++)
		  for (var j = 0; j < width; j++) {
                    var el = image[i][j][0]*298 + image[i][j][1]*587 
                                + image[i][j][2]*114;
                    
                    lessthens[i*width+j] = LessEqThan(32);
                    lessthens[i*width+j].in[0] <== el -  gs_image[i][j]*1000;
            		lessthens[i*width+j].in[1] <== 1000;	
                    lessthens[i*width+j].out * lessthens[i*height+j].out === 1;
        }
		
}
