from keras.models import model_from_json
import theano.tensor as T
from utils.readImgFile import readImg
from utils.crop import crop_detection
import os
import numpy as np
from keras.optimizers import SGD, Adam
from utils.ReadPascalVoc2 import generate_batch_data

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

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('Tiny_Yolo_weights.h5')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss=custom_loss)

vocPath= os.path.join(os.getcwd(),'dataset/train_val')
imageNameFile= os.path.join(vocPath,'imageNames.txt')
model.fit_generator(generate_batch_data(vocPath,imageNameFile,16),samples_per_epoch=4992,nb_epoch=5,verbose=1)

json_string = model.to_json()
model.save_weights('Tiny_Yolo_weights_iter2.h5')
