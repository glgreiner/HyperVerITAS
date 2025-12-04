from PIL import Image
import numpy as np
import json
import math

def imgToJSON( path, p ):
    pix = np.transpose(p, (2, 0, 1))
    
    width = len(pix[0][0])
    height = len(pix[0])
    flattenedImg = [pix[0].flatten(), pix[1].flatten(), pix[2].flatten()]
    # We make JSON image thing
    imgJSON = {   "rows": width,
                  "cols": height,
                  "R": flattenedImg[0].tolist(),
                  "G": flattenedImg[1].tolist(),
                  "B": flattenedImg[2].tolist()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(imgJSON, f, ensure_ascii=False, indent=4)


def makeCrop (path, pix, startX, startY, endX, endY):
    # This is looping over rows
    cropped = pix[startX:endX,startY:endY]
    # img = Image.fromarray(cropped)
    # img.save("cropped.jpg")
    imgToJSON(path, cropped)

# This function also returns the error term.
# We only blur a box
def makeBlur(path, pix, startX, startY, endX, endY):
    pass

# This function also returns the error term
def makeResize(path, pix, origX, origY, newX, newY):
    pass

# This function also returns the error term.
# We only redact a box
def makeRedact(path, pix, origX, origY, newX, newY):
    pass

# This function also returns the error term
# .30, .59, .11
def makeGray(path, pix):
    gray = []
    for i in range(len(pix)):
        gray.append([])
        for j in range(len(pix[0])):
            val = int(round(.3*pix[i][j][0]+.59*pix[i][j][1]+.11*pix[i][j][2]))
            gray[i].append([val,val,val])
    gray = np.array(gray)
    # print(gray)
    imgToJSON(path, gray)
# def testCropJSON( path, pix ):
#     width = len(pix[0][0])
#     height = len(pix[0])
#     flattenedImg = [pix[0].flatten(), pix[1].flatten(), pix[2].flatten()]
#     # We make JSON image thing
#     imgJSON = {   "rows": width,
#                   "cols": height,
#                   "R": flattenedImg[0].tolist(),
#                   "G": flattenedImg[1].tolist(),
#                   "B": flattenedImg[2].tolist()}
#     with open(path, 'w', encoding='utf-8') as f:
#         json.dump(imgJSON, f, ensure_ascii=False, indent=4)


# testCropChicken = Image.open('chicken.jpg')
# pix = np.array(testCropChicken)
# pix = np.array(testCropChicken)
# print(pix)
testSize = int(input("Image Size: "))
testFudge = testSize%2
# myImg = np.random.randint(255, size=(6632,4976,3))
print(int(testSize/2))
print(int(testFudge))
print(2**(int(testSize/2+testFudge)))
myImg = np.random.randint(255, size=(2**(int(testSize/2)),2**(int(testSize/2+testFudge)),3))
f = open(f"Veri{testSize}R.txt","w")
for i in range(len(myImg)):
    for j in range(len(myImg[0])):
        f.write(str(myImg[i][j][0]))
        f.write("\n")
f.close()
f = open(f"Veri{testSize}G.txt","w")
for i in range(len(myImg)):
    for j in range(len(myImg[0])):
        f.write(str(myImg[i][j][1]))
        f.write("\n")
f.close()
f = open(f"Veri{testSize}B.txt","w")
for i in range(len(myImg)):
    for j in range(len(myImg[0])):
        f.write(str(myImg[i][j][2]))
        f.write("\n")
f.close()
imgToJSON(f"Timings{testSize}.json", myImg )
makeGray(f"Gray{testSize}.json", myImg)
if testSize % 2 == 0:
    makeCrop(f"Crop{testSize}.json", myImg,0,0,2**(int((testSize/2))),2**(int(testSize/2-(1))))
else:
    makeCrop(f"Crop{testSize}.json", myImg,0,0,2**(int((testSize/2))),2**(int(testSize/2)))
print(len(myImg), len(myImg[0]))
print(int((testSize/2)))
print((int(testSize/2)))
# np.ones(2**11,2**11,3)