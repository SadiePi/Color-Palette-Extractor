# palette.py by Sadie Dotzler
# Attempts to extract a color palette from a given image using a fairly simple
# differentiable heuristic function to determine the prominence of any given
# color in the image.
#
# Note that I have approximately zero experience with color threory or similar
# fields. This was just an idea I had to solve this problem I'd been mulling
# for a while. I can't make any guaruntees of result quality
#
# I haven't tested how this changes if you use HSV instead of RGB; that might
# be interesting.

# The image to analyze
IMAGE = 'img3.jpg'

# Fairly self explanatory
COLOR_SCAN_STEP = 32
IMAGE_SCAN_STEP = 8

# How quickly the closeness of one color to another stops mattering
# Might be thought of as a measure of color heterogeneity
COLOR_PROXIMITY_DROPOFF_FACTOR = 1

from typing import Tuple
from PIL import Image
import numpy as np
from math import sqrt,exp,ceil
# kinda ridiculous that I need a library this bulky just for this
from scipy.ndimage import filters, morphology

# https://stackoverflow.com/questions/3986345
def detect_local_maxima(arr):
    # TODO Play around with changing 2 in this. I have no idea what it does
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    background = (arr==0)
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=0)
    detected_maxima = local_max ^ eroded_background
    return np.rot90(np.where(detected_maxima))

def mag(vec: Tuple[float,float,float]):
    return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2])

# The main heuristic function.
#
# Image the 3d color space cube. With each pixel color, take it to its place in
# that cube and raise the prominence at that point and around it, dropping off
# exponentially.
# 
# The resulting surface is diferrentiable, so we can later using a peak finding
# algorithm to find the most prominent colors and return those as the palette.  
def color_prominence(img, color):
    prominence = 0
    #img = img.convert('RGB')
    pixels = img.load()
    
    for x in range(0, img.width, IMAGE_SCAN_STEP):
        for y in range(0, img.height, IMAGE_SCAN_STEP):
            # An an exponentially decayed distance between this pixel and color
            prominence += exp(-mag(
                np.array(pixels[x,y])-color) * COLOR_PROXIMITY_DROPOFF_FACTOR)
    print(color, prominence) # print process so far
    return prominence


# Find the prominence of each color in the image
img = Image.open(IMAGE)
colorProminences = np.array(
    [[[color_prominence(img, np.array([r,g,b]))
       for r in range(0, 256, COLOR_SCAN_STEP)]
       for g in range(0, 256, COLOR_SCAN_STEP)]
       for b in range(0, 256, COLOR_SCAN_STEP)])

# Find the highest prominence colors to determine the palette
palette = np.array([m * COLOR_SCAN_STEP 
                    for m in detect_local_maxima(colorProminences)])

# Display Palette
windowSize = ceil(sqrt(len(palette)))
view = Image.new('RGB', (windowSize, windowSize))
for i in range(len(palette)):
    x, y = i%windowSize, i//windowSize
    view.putpixel((x, y), tuple(palette[i]))
view.resize((250, 250), Image.NEAREST).show()

# Print Palette
print("\nResult:")
print(np.array(
    list((list(c), 
        colorProminences[c[0]//COLOR_SCAN_STEP]
        [c[1]//COLOR_SCAN_STEP]
        [c[2]//COLOR_SCAN_STEP])
    for c in palette),
    dtype=object)
)
