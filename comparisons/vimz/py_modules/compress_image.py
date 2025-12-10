import tkinter as tk
from tkinter import filedialog
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


VESTA_PRIME = 28948022309329048855892746252171976963363056481941647379679742748393362948097


def get_image_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def compress(image_in):
    array_in = np.array(image_in).tolist()
    output_array = []
    # print(len(array_in), len(array_in[0]), len(array_in[0][0]))

    for i in range(0, len(array_in)):
        row = []
        hexValue = ''
        for j in range(0, len(array_in[i])):
            if np.isscalar(array_in[i][j]):
                hexValue = hex(int(array_in[i][j]))[2:].zfill(6) + hexValue
            else:
                for k in range(0, 3):
                    hexValue = hex(int(array_in[i][j][k]))[2:].zfill(2) + hexValue
            if j % 10 == 9:
                row.append("0x" + hexValue)
                hexValue = ''
        output_array.append(row)
    return output_array

def compress_image(image_path):
    with Image.open(image_path) as image:
        return compress(image)

# Get the image path using Tkinter file dialog
image_path = get_image_path()

if image_path:
    # Example usage
    
    # Crop the image and save it
    compressed_original_image = compress_image(image_path)
    out = {
        "original": compressed_original_image,
    }
    print("Image compressed successfully.")

    transformation = input("Input Transformation\n")

    output_path = f"transformation_{transformation}.json"

    with open(output_path, 'w') as fp:
        json.dump(out, fp, indent=4)
    print("Image data dumped successfully.")
else:
    print("No image selected.")
