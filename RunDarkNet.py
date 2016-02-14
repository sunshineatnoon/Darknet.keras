import os
import numpy as np
from utils.readImgFile import readImg
from utils.DarkNet import ReadDarkNetWeights
from utils.crop import crop

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import theano
from keras.layers.core import Flatten, Dense, Activation, Reshape

theano.config.optimizer = 'fast_compile'
def SimpleNet(darkNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(16, 3, 3, weights=[darkNet.layers[1].weights,darkNet.layers[1].biases],border_mode='valid',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Use a for loop to replace all manually defined layers
    for i in range(3,17):
        l = darkNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            if(i == 12):
                model.add(MaxPooling2D(pool_size=(2, 2),border_mode='custom'))
            else:
                model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        elif(l.type == "AVGPOOL"):
            model.add(AveragePooling2D(pool_size=(4,4),strides=(4,4)))
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "SOFTMAX"):
            model.add(Activation('softmax'))
        else:
            print "Error: Unknown Layer Type",l.type
    return model

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

#image = readImg(os.path.join(os.getcwd(),'images/dog.file'))
image = crop(os.path.join(os.getcwd(),'images/eagle.jpg'))
image = np.expand_dims(image, axis=0)

darkNet = ReadDarkNetWeights(os.path.join(os.getcwd(),'weights/darknet.weights'))
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


model = SimpleNet(darkNet)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

image = readImg(os.path.join(os.getcwd(),'Yolo_dog.img'),h=448,w=448)
image = np.expand_dims(image, axis=0)
act = get_activations(model, 13,image)
print act[0]
'''
out = model.predict(image)
prediction = out[0]

#output first five predicted labels
f = open(os.getcwd()+'/images/shortnames.txt')
lines = f.readlines()
for i in range(len(lines)): lines[i] = lines[i].strip("\n")

indices = prediction.argsort()[-10:][::-1]
for i in indices:
    print lines[i],": %f"%prediction[i]
'''
