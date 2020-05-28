import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("montain.jpg")
im_array = np.array(im)

# print(im_array.shape[0])

def histogram(im):
    h = np.zeros(255)
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            val = im[row, col]
            h[val] += 1
    return h

x = histogram(im_array)
# print(x)
# plt.bar(range(len(x)), x)
# plt.show()
# print(x.shape[0])

def Probability(y,im):
    h1 = np.zeros(255)
    for x in range(y.shape[0]):
        h1[x] = y[x]/(im.shape[0]*im.shape[1])
    return h1

def add_up(y):
    h2 = np.zeros(255)
    for x in range(y.shape[0]):
        for n in range(x):
            h2[x] += y[n]
    return h2

x1 = Probability(x,im_array)
x2 = add_up(x1)
x3= x2*255

def map(x,im):
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            im[row, col] = x[im[row,col]]
    return im

x4 = map(x3,im_array)
im=Image.fromarray(x4.astype(np.uint8))
im.show()

# plt.bar(range(len(x3)), x3)
# plt.show()