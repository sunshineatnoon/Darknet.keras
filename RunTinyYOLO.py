import os
import numpy as np

from utils.readImgFile import readImg
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.crop import crop

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

import theano

def SimpleNet(yoloNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(1,1),input_shape=(3,448,448)))
    model.add(Convolution2D(16, 3, 3, weights=[yoloNet.layers[1].weights,yoloNet.layers[1].biases],border_mode='valid',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Use a for loop to replace all manually defined layers
    for i in range(3,20):
        l = yoloNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        elif(l.type == "FLATTEN"):
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
            model.add(LeakyReLU(alpha=0.1))
        else:
            print "Error: Unknown Layer Type",l.type
    return model

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

image = readImg(os.path.join(os.getcwd(),'images/Yolo_dog.img'),h=448,w=448)
#image = crop(os.path.join(os.getcwd(),'images/eagle.jpg'))
image = np.expand_dims(image, axis=0)

yoloNet = ReadTinyYOLONetWeights(os.path.join(os.getcwd(),'weights/yolo-tiny.weights'))
#reshape weights in every layer
for i in range(yoloNet.layer_number):
    l = yoloNet.layers[i]
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


model = SimpleNet(yoloNet)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#out = model.predict(image)

for i in range(34,38):
    out = get_activations(model,i,image)
    print out.shape
    print out[0,0:10]
