from PIL import Image
import numpy as np
from scipy import misc
import os

def crop(imPath):
    im = Image.open(imPath)
    im = im.resize((256,256),Image.ANTIALIAS)

    #central crop 224,224
    width, height = im.size   # Get dimensions
    new_width = 224
    new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))
    image_array = np.array(im)
    image_array = np.rollaxis(image_array,2,0)
    image_array = image_array/255.0
    image_array = image_array * 2.0 - 1.0
    return image_array
