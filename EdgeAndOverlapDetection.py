import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import cv2
import os
import imageio.v2 as imageio
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure, morphology, restoration, segmentation, transform, util)
import plotly.express as px
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as im

#Opening save path and tiff file
ZStack_path = r"C:\Image_Analysis\James\2023-08-01\Plant1_2023_08_01\Plant1_2023_08_01\Plant1_2023_08_01_MMStack_Pos01.ome.tif"
tmp_path = r"C:\Image_Analysis\tmp"

#Changing file directory and counting the number of layers in the tiff along with its x/y pixel count 
os.chdir(tmp_path)
image = cv2.imread(ZStack_path)
img = io.imread(ZStack_path)
slices, x, y  = (img.shape)

#Clears tmp file
test = os.listdir(tmp_path)
for images in test:
    if images.endswith(".jpeg"):
        os.remove(os.path.join(tmp_path, images))

#Converts image and applies a gaussian blur to image
for i in range(slices):
    print (i)
    normalizedImg = np.zeros((2000, 2000))
    normalizedImg = cv2.normalize(img[i,:,:],  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(i)+'v1.jpeg',normalizedImg)
    image_Gaus = Image.open(str(i)+'v1.jpeg') 
    image_Guas = image_Gaus.filter(ImageFilter.GaussianBlur(radius=.0))
    image_Guas.save(str(i)+'v1.jpeg')

#Loads blurred image, uses Canny to identify edges in moss
for i in range(slices):
    print (i)
    normIMG = cv2.imread(str(i)+'v1.jpeg', cv2.IMREAD_GRAYSCALE)
    loaded_image = cv2.cvtColor(normIMG,cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)
    edged_image = cv2.Canny(gray_image, 100, 256)
    cv2.imwrite(str(i)+'v2.jpeg',edged_image)

    #Identifying overlapping regions in the cell
    if i >= 1:
        Overlap1 = cv2.imread(str(i)+'v2.jpeg', cv2.IMREAD_GRAYSCALE)
        Overlap2 = cv2.imread(str(i - 1)+'v2.jpeg', cv2.IMREAD_GRAYSCALE)
        bitwiseand = cv2.bitwise_and(Overlap1, Overlap2)
        plt.imshow(bitwiseand)
        cv2.imwrite(str(i)+'overlap.png', bitwiseand)
        cv2.imwrite(str(i)+'overlap_tests.png', bitwiseand)
        bitwiseand = Image.open(str(i)+'overlap.png')
        clrs = bitwiseand.getcolors()
        #Compares two images for overlay 
        if i >= 2:
            Overlap1 = cv2.imread(str(i)+'overlap_tests.png', cv2.IMREAD_GRAYSCALE)
            Overlap2 = cv2.imread(str(i-1)+'overlap_tests.png', cv2.IMREAD_GRAYSCALE)
            bitwiseand = cv2.bitwise_and(Overlap1, Overlap2)
            plt.imshow(bitwiseand)
            cv2.imwrite(str(i)+'overlap.png', bitwiseand)
            bitwiseand = Image.open(str(i)+'overlap.png')
            clrs = bitwiseand.getcolors()     
#combines the overlays
for i in range(slices):
    matrix = im.imread(str(i)+'v2.jpeg')
    if i >= 1:
        matrix2 = im.imread(str(i-1)+'v2.jpeg')
        output = np.add(matrix,matrix2)
        cv2.imwrite(str(i)+'v2.jpeg', output)
#Divides all branch overlay by 5 for reduced brightness
matrix = im.imread(str(i)+'v2.jpeg')
output = np.divide(matrix, 5)
cv2.imwrite('Final.jpeg', output)

#Combines the total .2 intensity brightness overlay with overlapping branches at full brightness
for i in range(slices-1):
    matrix2 = Image.open('Final.jpeg')
    if i != 0:
        matrix = Image.open(str(i+1)+'overlap_tests.png')
        result = ImageChops.lighter(matrix, matrix2)
        Product = result.save('Final.jpeg')
    print (i)
    
