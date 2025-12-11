from PIL import Image
import numpy as np
import json
import math

def imgToJSON(path, p):
    pix = np.transpose(p, (2, 0, 1))
    
    width = len(pix[0][0])
    height = len(pix[0])
    flattenedImg = [pix[0].flatten(), pix[1].flatten(), pix[2].flatten()]

    # we make an object storing the image in JSON format
    imgJSON = {"rows": width,
               "cols": height,
               "R": flattenedImg[0].tolist(),
               "G": flattenedImg[1].tolist(),
               "B": flattenedImg[2].tolist()}
    
    # dump the obeject into a JSON file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(imgJSON, f, ensure_ascii=False, indent=4)

def makeCrop (path, pix, startX, startY, endX, endY):
    # take only the cropped region of the image
    cropped = pix[startX:endX,startY:endY]

    # save the cropped image as a JSON file
    imgToJSON(path, cropped)

def makeGray(path, pix):
    # this will store the grayscaled image
    gray = []
    for i in range(len(pix)):
        gray.append([])
        for j in range(len(pix[0])):
            # compute grayscale value (Y = 0.3*R + 0.59*G + 0.11*B)
            val = int(round(.3*pix[i][j][0]+.59*pix[i][j][1]+.11*pix[i][j][2]))
            gray[i].append([val,val,val])
    gray = np.array(gray)

    # save the grayscaled image as a JSON file
    imgToJSON(path, gray)

def generateImage(size):
    # create an image of size 2^size pixels
    # creates grayscaled and cropped variants as well
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

    # make JSON file for the image
    imgToJSON(f"Timings{size}.json", myImg)

    # make grayscale image
    makeGray(f"Gray{size}.json", myImg)

    # create cropped image (half of image)
    if size % 2 == 0:
        makeCrop(f"Crop{size}.json", myImg,0,0,2**(dim1),2**(int(dim1-(1))))
    else:
        makeCrop(f"Crop{size}.json", myImg,0,0,2**(dim1),2**(dim1))

    print(f"generation complete for size 2^{size}!\n")

def generateImages(start, end):
    print("Generating Images...\n")
    for i in range(start, end+1):
        generateImage(i)
    print("All Images Generated!")


generateImages(19, 25)
