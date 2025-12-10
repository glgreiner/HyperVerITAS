from PIL import Image
import numpy as np
import math

def generateImage(size):
    # create an image of size 2^size pixels
    print(f"generating image data for size 2^{size}")

    parity = size % 2

    # these will be the dimensions of the image
    dim1 = int(size/2)
    dim2 = int(size/2 + parity)

    # create a image of random pixel values 
    myImg = np.random.randint(255, size=(2**(dim1),2**(dim2),3))

    # store red channel in a text file
    f = open(f"Veri{size}R.txt","w")
    for i in range(len(myImg)):
        for j in range(len(myImg[0])):
            f.write(str(myImg[i][j][0]))
            f.write("\n")
    f.close()

    # store green channel in a text file
    f = open(f"Veri{size}G.txt","w")
    for i in range(len(myImg)):
        for j in range(len(myImg[0])):
            f.write(str(myImg[i][j][1]))
            f.write("\n")
    f.close()

    # store blue channel in a text file
    f = open(f"Veri{size}B.txt","w")
    for i in range(len(myImg)):
        for j in range(len(myImg[0])):
            f.write(str(myImg[i][j][2]))
            f.write("\n")
    f.close()

    print(f"generation complete for size 2^{size}!\n")

def generateImages(start, end):
    print("Generating Images...\n")
    for i in range(start, end+1):
        generateImage(i)
    print("All Images Generated!")


generateImages(19, 20)
