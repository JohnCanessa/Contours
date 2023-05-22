# **** ****
import numpy as np                          # for array handling
from matplotlib import pyplot as plt        # for plotting

# **** ****
import skimage.io                           # for reading images
from skimage import measure                 # for finding contours
from skimage import color                   # for gray2rgb
from skimage.util import view_as_blocks     # for block processing

# **** ****
from PIL import Image                       # for reading images


# **** read shapes.png ****
shapes = skimage.io.imread("./images/shapes.png")

# **** display shapes.shape 
#      (PNG images have an additional channel for opacity or alpha) ****
print('shapes.shape = ', shapes.shape)

# **** display shapes ****
plt.imshow(shapes)
plt.title("shapes")
plt.show()


# **** copy the RGB PNG image ****
shapes = Image.open('./images/shapes.png')

# **** convert RGBA to RGB ****
shapes = shapes.convert('RGB')

# **** convert RGB PNG image to grayscale ****
shapes = color.rgb2gray(shapes)

# **** display shapes.shape ****
print('shapes.shape = ', shapes.shape)

# **** display shapes ****
plt.imshow(shapes, cmap='gray')
plt.title("shapes - gray")
plt.show()


# **** find contours
#      with level 0.5 we get 4 contours
#      with level 0.9 we get 5 contours ****
contours = measure.find_contours(   shapes,         # image to process
                                    0.9)            # 0.5 is the level value to find contours at

# **** count contours ****
count = len(contours)

# **** display count of contours ****
print("count: ", count)


# **** create a figure and set of subplots ****
fig, axes = plt.subplots(   1,                      # number of rows
                            2,                      # number of columns
                            figsize=(16, 12),       # figure size in inches
                            sharex=True,            # share x axis
                            sharey=True)            # share y axis

# **** ****
ax = axes.ravel()                                   # flatten axes

# **** display original shapes image ****
ax[0].imshow(   shapes,
                cmap='gray')
ax[0].set_title('Original image',                   # set title
                fontsize=20)                        # set font size

# **** display shape images with contours ****
ax[1].imshow(   shapes,
                cmap='gray',
                interpolation='nearest')
ax[1].set_title('Contour',                          # set title
                fontsize=20)                        # set font size

# **** ****
for n, contour in enumerate(contours):
    ax[1].plot( contour[:, 1],                       # x axis
                contour[:, 0],                       # y axis
                linewidth=5)                         # line width

# **** ****
plt.show()                                          # display the figure


# **** ****
A, B = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]

# **** generate complex image using NumPy ****
complex_image = np.sin(np.exp((np.sin(A)**3 + np.cos(B)**3)))

# **** display complex_image.shape ****
print('complex_image.shape: ', complex_image.shape)

# **** display complex_image ****
plt.imshow(complex_image, cmap='gray')              # display the image in gray scale
plt.title("complex_image")                          # set title
plt.show()                                          # display the image


# **** find contours ****
contours = measure.find_contours(   complex_image,  # image to process
                                    0.7)            # 0.7 is the level value to find contours

# **** count contours ****
count = len(contours)

# **** display count of contours ****
print("count: ", count)


# **** create a figure and set of subplots ****
fig, axes = plt.subplots(figsize=(12, 8))           # figure size in inches

# **** ****
axes.imshow(    complex_image,
                cmap='gray',
                interpolation='nearest')
axes.set_title('Contour',                           # set title
                fontsize=20)                        # set font size

# **** ****
for n, contour in enumerate(contours):
    axes.plot( contour[:, 1],                       # x axis
                contour[:, 0],                      # y axis
                linewidth=5)                        # line width

# **** ****
plt.show()                                          # display the figure