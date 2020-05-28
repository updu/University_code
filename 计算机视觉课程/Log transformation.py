import numpy as np
from PIL import Image

im = Image.open("city.jpeg")
im_array = np.asarray(im)

print(im_array.shape)
value_array = np.zeros(im_array.shape)

print(value_array)

def Log_transformation(im):

    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            value_array[row,col] = np.floor((255/np.log2(256))*np.log2(1+im[row, col]))
    return value_array
x = Log_transformation(im_array )
im=Image.fromarray(x.astype(np.uint8))
im.show()


# print(x)
# plt.imsave('0.png',x)
