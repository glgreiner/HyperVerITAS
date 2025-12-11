
pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/comparators.circom";

template Crop(height, width, cropped_height, cropped_width, cropped_height_start, cropped_width_start) {
    signal input image[height][width][3];
    signal input cropped_image[cropped_height][cropped_width][3];
    
    component isequals[cropped_height*cropped_width*3];
    for (var i = 0; i <  cropped_height; i++)
		for (var j = 0; j < cropped_width; j++) 
			for (var k = 0; k < 3; k++) 
            {   
                var el = i*(cropped_width*3)+j*3+k;
                isequals[el] = IsEqual();
                isequals[el].in[0] <== cropped_image[i][j][k];
                isequals[el].in[1] <== image[cropped_height_start + i][cropped_width_start + j][k];	
                isequals[el].out * isequals[el].out === 1;
            }
				

}
/** OLD CROP

template Crop(height, width, cropped_height, cropped_width, cropped_height_start, cropped_width_start) {
    signal input image[height][width][3];
    signal input cropped_image[cropped_height][cropped_width][3];

    for (var i = 0; i <  cropped_height; i++)
		for (var j = 0; j < cropped_width; j++) 
			for (var k = 0; k < 3; k++) 
				cropped_image[i][j][k] === image[cropped_height_start + i][cropped_width_start + j][k];	

}
*/ 
