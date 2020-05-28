import numpy as np
from PIL import Image

im = Image.open("lena.jpg").convert("L")
# im = Image.open("license_plate.png").convert("L")
im_array = np.asarray(im)
# print(im_array)

f1 = np.array([[1,1,1],
      [0,0,0],
      [-1,-1,-1]])
f2 = np.array([[-1,0,1],
      [-1,0,1],
      [-1,0,1]])

def edge_detection(f,im_array):
    im_new = np.zeros((im_array.shape))

    for x in range(1,im_array.shape[0]-1):
        for y in range(1,im_array.shape[1]-1):
            mul_array = im_array[x-1:x +2,y-1:y+2]
            # print(mul_array,f1)
            im_new[x, y] = (np.multiply(mul_array,f)/3).sum()
    return  im_new
im_vertical = edge_detection(f1,im_array)
im_horizontal = edge_detection(f2,im_array)
# print(im_new)
im1=Image.fromarray(im_vertical.astype(np.uint8))
im1.show()
# im2=Image.fromarray(im_horizontal.astype(np.uint8))
# im2.show()