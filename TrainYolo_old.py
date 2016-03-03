import os
import numpy as np

from utils.readImgFile import readImg
from utils.DarkNet import ReadDarkNetWeights
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.timer import Timer
from utils.ReadPascalVoc2 import generate_batch_data

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras.utils.visualize_util import plot
import keras

from math import pow
import theano
import theano.tensor as T
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join


def SimpleNet(darkNet,yoloNet):
    '''
    Args:
      darkNet: dark net weights, to initialize the weights of the first 13 layers
      yoloNet: yolo net, only need the structure parameters here
    Returns:
      model: A keras model which defines Tiny Yolo Net, with its first 13 layers' weights initialized by darknet
    '''
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
        #model.add(Activation('sigmoid'))
    return model

def get_activations(model, layer, X_batch):
    '''
    Args:
      model: keras model which defines net architecture
      layer: which layer's activation you want
      X_batch: input data for the forward pass
    Returns:
      activations: The activations in layer layer
    '''
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

def custom_loss(y_true,y_pred):
    '''
    Args:
      y_true: Ground Truth output
      y_pred: Predicted output
      The forms of these two vectors are:
      ######################################
      ## x,y,h,w,p1,p2,...,p20,objectness ##
      ######################################
    Returns:
      The loss caused by y_pred
    '''
    y1 = y_pred
    y2 = y_true
    loss = 0.0

    scale_vector = []
    scale_vector.extend([2]*4)
    scale_vector.extend([1]*20)
    scale_vector = np.reshape(np.asarray(scale_vector),(1,len(scale_vector)))

    for i in range(49):
        y1_piece = y1[:,i*25:i*25+24]
        y2_piece = y2[:,i*25:i*25+24]

        y1_piece = y1_piece * scale_vector
        y2_piece = y2_piece * scale_vector

        loss_piece = T.sum(T.square(y1_piece - y2_piece),axis=1)
        loss = loss + loss_piece * y2[:,i*25+24]
        loss = loss + T.square(y2[:,i*25+24] - y1[:,i*25+24])

    loss = T.sum(loss)
    return loss

class LossHistory(keras.callbacks.Callback):
    '''
    Use LossHistory to record loss
    '''
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def draw_loss_func(loss):
    print 'Loss Length is:',len(loss)
    loss = loss[150:]
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.savefig('loss.jpg')

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
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss=custom_loss)
plot(model, to_file='model.png')
'''
image = readImg(os.path.join(os.getcwd(),'Yolo_dog.img'),h=448,w=448)
image = np.expand_dims(image, axis=0)
act = get_activations(model, 13,image)
print act[0]
'''
vocPath= os.path.join(os.getcwd(),'dataset/train_val')
imageNameFile= os.path.join(os.getcwd(),'07_12.txt')
history = LossHistory()

print 'Start Training...'
model.fit_generator(generate_batch_data(vocPath,imageNameFile,16,9952),samples_per_epoch=9952,nb_epoch=15,verbose=1,callbacks=[history])

print 'Saving loss graph'
draw_loss_func(history.losses)

print 'Saving weights and model architecture'
json_string = model.to_json()
open('Tiny_Yolo_Architecture.json','w').write(json_string)
model.save_weights('Tiny_Yolo_weights_iter4.h5',overwrite=True)
