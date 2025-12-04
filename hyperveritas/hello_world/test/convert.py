from PIL import Image
import numpy as np
import json
import math
import sys

filename = sys.argv[1]
imageSize = int(sys.argv[2])

def convert_three_to_one(path_from, path_to):
    with open(path_from, 'r') as file:
        img = json.load(file)
        r_vals = img["R"]
        rows = img["rows"]
        cols = img["cols"]
        
    f = open(path_to,"w")
    f.write(str(rows))
    f.write("\n")
    f.write(str(cols))
    f.write("\n")
    for i in range(len(r_vals)):
        f.write(str(r_vals[i]))
        f.write("\n")
    f.close()


convert_three_to_one(f"{filename}{imageSize}.json", f"{filename}{imageSize}R.txt")
