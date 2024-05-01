import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import cv2
import os
import imageio.v2 as imageio
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform, util)
import plotly.express as px
from PIL import Image, ImageFilter
import matplotlib as mpl 
import matplotlib.pyplot as plt 

ZStack_path = r"C:\Image_Analysis\James\2023-08-01\Plant1_2023_08_01\Plant1_2023_08_01\Plant1_2023_08_01_MMStack_Pos01.ome.tif"
tmp_path = r"C:\Image_Analysis\tmp"

os.chdir(tmp_path)
image = cv2.imread(ZStack_path)
img = io.imread(ZStack_path)
slices, x, y  = (img.shape)

test = os.listdir(tmp_path)
for images in test:
    if images.endswith(".jpeg"):
        os.remove(os.path.join(tmp_path, images))


for i in range(slices):
    print (i)
    normalizedImg = np.zeros((2000, 2000))
    normalizedImg = cv2.normalize(img[i,:,:],  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(i)+'v1.jpeg',normalizedImg)
    image_Gaus = Image.open(str(i)+'v1.jpeg') 
    image_Guas = image_Gaus.filter(ImageFilter.GaussianBlur(radius=.5))
    image_Guas.save(str(i)+'v1.jpeg')

for i in range(slices):
    print (i)
    normIMG = cv2.imread(str(i)+'v1.jpeg')
    loaded_image = cv2.cvtColor(normIMG,cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)
    edged_image = cv2.Canny(gray_image, 150, 256)
    cv2.imwrite(str(i)+'v2.jpeg',edged_image)

 
EdgeInitial = cv2.imread(str(0)+'v2.jpeg')
Combine = slices

for i in range(Combine):
    EdgeInitial = Image.open(str(i)+'v2.jpeg')
    img = EdgeInitial.convert("RGB")
    d = img.getdata()
    new_image = []
    for item in d:
        if item[0] in list(range(20, 256)):
            red = -255 + ((255/Combine) * i)
            blue = 255 - (255/Combine) * i
            new_image.append((int(red), 0, int(blue)))
        else:
            new_image.append(item)
    img.putdata(new_image)
    img.save(str(i) + "v3.jpeg")
    EdgeInitial = Image.open(str(i)+'v3.jpeg')
    if i == 0:
        NextEdge = Image.open(str(i)+'v3.jpeg')
    else:
        NextEdge = Image.open('result.jpeg')
    print (i)

    result = ImageChops.lighter(EdgeInitial, NextEdge)
    Product = result.save('result.jpeg')

norm = mpl.colors.Normalize(vmin=0, vmax=slices) 
slice_1 = slices-1
cmap = plt.get_cmap('bwr', slices-1) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
sm.set_array([]) 
plt.colorbar(sm, ticks=np.linspace(0, slice_1, slices)) 
plt.show() 

image = cv2.imread('result.jpeg')
alpha = 3
beta = 0

adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imwrite('result.jpeg',adjusted)


