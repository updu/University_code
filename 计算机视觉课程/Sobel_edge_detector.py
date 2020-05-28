import numpy as np
from PIL import Image

# im = Image.open("lena.jpg").convert("L")
im = Image.open("license_plate.png").convert("L")
im_array = np.asarray(im)

f1 = np.array([[1,0,-1],
                [2,0,-2],
                [1,0,-1]])
f2 = np.array([[1,2,1],
              [0,0,0],
              [-1,-2,-1]])

def edge_detection(f,im_array):
    im_new = np.zeros((im_array.shape))

    for x in range(1,im_array.shape[0]-1):
        for y in range(1,im_array.shape[1]-1):
            mul_array = im_array[x-1:x +2,y-1:y+2]
            # print(mul_array,f1)
            im_new[x, y] = (np.multiply(mul_array,f)).sum()
    return  im_new

def map(array):
    array_new = np.zeros(array.shape)
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            array_new[x,y] = 255/array.max()*array[x,y]
    return array_new

Gx = edge_detection(f1,im_array)
Gy = edge_detection(f1,im_array)

G1 = np.sqrt(np.square(Gx)+np.square(Gy))

G = map(G1)
print(G)
im=Image.fromarray(G.astype(np.uint8))
im.show()