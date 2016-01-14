import os
import numpy as np
from utils.readImgFile import readImg
from utils.PythonReader import ReadGoogleNetWeights
from utils.crop import crop

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import theano
from keras.layers.core import Flatten, Dense, Activation

def SimpleNet(googleNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(3,3),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 7, 7, weights=[googleNet.layers[1].weights,googleNet.layers[1].biases],border_mode='valid',subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Use a for loop to replace all manually defined layers
    for i in range(3,27):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif(l.type == "AVGPOOL"):
            model.add(AveragePooling2D(pool_size=(7,7),strides=(7,7)))
            #In this particular model, feature output by avg pooling needs to be flattened. TODO:Use a better way to indicate there is a flatten layer
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
            model.add(LeakyReLU(alpha=0.1))
        else:
            print "Error: Unknown Layer Type"
    model.add(Activation('softmax'))
    return model

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

#image = readImg(os.path.join(os.getcwd(),'images/dog.file'))
image = crop(os.path.join(os.getcwd(),'images/dog.jpg'))
image = np.expand_dims(image, axis=0)

googleNet = ReadGoogleNetWeights(os.path.join(os.getcwd(),'weights/extraction.weights'))
#reshape weights in every layer
for i in range(googleNet.layer_number):
    l = googleNet.layers[i]
    if(l.type == 'CONVOLUTIONAL'):
        weight_array = l.weights
        n = weight_array.shape[0]
        weight_array = weight_array.reshape((n//(l.size*l.size),(l.size*l.size)))[:,::-1].reshape((n,))
        weight_array = np.reshape(weight_array,[l.n,l.c,l.size,l.size])
        l.weights = weight_array
    if(l.type == 'CONNECTED'):
        weight_array = l.weights
        #weight_array = np.reshape(weight_array,[l.input_size,l.output_size])
        weight_array = np.reshape(weight_array,[l.output_size,l.input_size])
        weight_array = weight_array.transpose()
        l.weights = weight_array


model = SimpleNet(googleNet)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(image)
prediction = out[0]

#output first five predicted labels
f = open(os.getcwd()+'/images/shortnames.txt')
lines = f.readlines()
for i in range(len(lines)): lines[i] = lines[i].strip("\n")

indices = prediction.argsort()[-10:][::-1]
for i in indices:
    print lines[i],": %f"%prediction[i]
