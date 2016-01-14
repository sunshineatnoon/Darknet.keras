import numpy as np
import os

def readImg(imgPath):
    dt = np.dtype("float32")
    testArray = np.fromfile(imgPath,dtype=dt)

    image = np.reshape(testArray,[3,224,224])
    return image
