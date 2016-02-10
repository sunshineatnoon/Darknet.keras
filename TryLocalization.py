import os
import numpy as np

from utils.readImgFile import readImg
from utils.DarkNet import ReadDarkNetWeights
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.crop import crop
from utils.timer import Timer
from utils.ReadPascalVoc2 import generate_batch_data

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout

from math import pow
import theano
import theano.tensor as T

from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join


def SimpleNet(darkNet,yoloNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(1,1),input_shape=(3,448,448)))
    model.add(Convolution2D(16, 3, 3, weights=[darkNet.layers[1].weights,darkNet.layers[1].biases],border_mode='valid',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #initialize first 13 layers using weights of darknet
    for i in range(3,14):
        l = darkNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

    for i in range(14,yoloNet.layer_number):
        l = yoloNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, init='he_normal',border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        elif(l.type == "FLATTEN"):
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size,init='he_normal'))
        elif(l.type == "LEAKY"):
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "DROPOUT"):
            model.add(Dropout(0.5))
        else:
            print "Error: Unknown Layer Type",l.type
        model.add(Activation(sigmoid))
    return model


darkNet = ReadDarkNetWeights(os.path.join(os.getcwd(),'weights/darknet.weights'))
yoloNet = ReadTinyYOLONetWeights(os.path.join(os.getcwd(),'weights/yolo-tiny.weights'))
#reshape weights in every layer
for i in range(darkNet.layer_number):
    l = darkNet.layers[i]
    if(l.type == 'CONVOLUTIONAL'):
        weight_array = l.weights
        n = weight_array.shape[0]
        weight_array = weight_array.reshape((n//(l.size*l.size),(l.size*l.size)))[:,::-1].reshape((n,))
        weight_array = np.reshape(weight_array,[l.n,l.c,l.size,l.size])
        l.weights = weight_array
    if(l.type == 'CONNECTED'):
        weight_array = l.weights
        weight_array = np.reshape(weight_array,[l.input_size,l.output_size])
        l.weights = weight_array

model = SimpleNet(darkNet,yoloNet)

#sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mean_squared_error')

vocPath= os.path.join(os.getcwd(),'dataset/train_val')
imageNameFile= os.path.join(vocPath,'imageNames.txt')
model.fit_generator(generate_batch_data(vocPath,imageNameFile),samples_per_epoch=5000,nb_epoch=5,verbose=1)
#image = readImg(os.path.join(os.getcwd(),'Yolo_dog.img'),h=448,w=448)
#image = np.expand_dims(image,axis=0)
#out = model.predict(image)
#print out.shape
json_string = model.to_json()
open('Tiny_Yolo_Architecture.json','w').write(json_string)
model.save_weights('Tiny_Yolo_weights.h5')
