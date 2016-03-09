import os
import numpy as np

from utils.readImgFile import readImg
from utils.DarkNet import ReadDarkNetWeights
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.timer import Timer
from utils.ReadPascalVoc2 import generate_batch_data
from utils.MeasureAccuray import MeasureAcc

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from keras.utils.visualize_util import plot
from keras.models import model_from_json

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

    #loss = T.sum(loss)
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

class TestAcc(keras.callbacks.Callback):
    '''
    calculate test accuracy after each epoch
    '''

    def on_epoch_end(self,epoch, logs={}):
        #Save check points
        filepath = 'weights'+str(epoch)+'.hdf5'
        print('Epoch %03d: saving model to %s' % (epoch, filepath))
        self.model.save_weights(filepath, overwrite=True)

        #Test train accuracy, only on 2000 samples
        vocPath = os.path.join(os.getcwd(),'dataset/train_val')
        imageNameFile = os.path.join(os.getcwd(),'dataset/train_val/shortlist.txt')
        sample_number = 200
        acc,re = MeasureAcc(self.model,sample_number,vocPath,imageNameFile)
        print 'Accuracy and recall on train data is: %3f,%3f'%(acc,re)

        #Test test accuracy, only on 200 samples
        vocPath = os.path.join(os.getcwd(),'dataset/VOC2012')
        imageNameFile = os.path.join(os.getcwd(),'shortlist_train.txt')
        sample_number = 200
        acc,re = MeasureAcc(self.model,sample_number,vocPath,imageNameFile)
        print 'Accuracy and recall on test data is: %3f,%3f'%(acc,re)

class printbatch(keras.callbacks.Callback):
    def on_batch_end(self,epoch,logs={}):
        print(logs)

def draw_loss_func(loss):
    print 'Loss Length is:',len(loss)
    loss = loss[100:]
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.savefig('loss.jpg')

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('Tiny_Yolo_weights_07_12_26720.h5')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss=custom_loss)

vocPath= os.path.join(os.getcwd(),'dataset/VOC2012')
imageNameFile= os.path.join(vocPath,'trainval.txt')

history = LossHistory()
testAcc = TestAcc()

print 'Start Training...'
batch_size = 16
model.fit_generator(generate_batch_data(vocPath,imageNameFile,batch_size=batch_size,sample_number=16551),samples_per_epoch=33102,nb_epoch=5,verbose=1,callbacks=[history,testAcc])

json_string = model.to_json()
model.save_weights('Tiny_Yolo_weights_07_12_26720_iter2.h5')
