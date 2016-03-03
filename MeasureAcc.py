from keras.models import model_from_json
import theano.tensor as T

from utils.readImgFile import readImg
from utils.crop import crop_detection
from utils.ReadPascalVoc2 import prepareBatch

import os
import numpy as np


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

def Acc(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    object_num = 0

    count = 0
    for image in imageList:
	    count += 1
        if(count % 500 == 0):
            print 'Image number:', count
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]

        for i in range(49):
            preds = out[i*25:(i+1)*25]
            if(preds[24] > thresh):
                object_num += 1
            	row = i/7
            	col = i%7
                '''
            	centerx = 64 * col + 64 * preds[0]
            	centery = 64 * row + 64 * preds[1]

            	h = preds[2] * preds[2]
            	h = h * 448.0
            	w = preds[3] * preds[3]
            	w = w * 448.0

            	left = centerx - w/2.0
            	right = centerx + w/2.0
            	up = centery - h/2.0
            	down = centery + h/2.0

            	if(left < 0): left = 0
            	if(right > 448): right = 447
            	if(up < 0): up = 0
            	if(down > 448): down = 447
                '''
            	class_num = np.argmax(preds[4:24])

                #Ground Truth
                box = image.boxes[row][col]
                if(box.has_obj):
                    for obj in box.objs:
                        true_class = obj.class_num
                        if(true_class == class_num):
                            correct += 1
	                    break


    return correct*1.0/object_num

def Recall(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    obj_num = 0
    count = 0
    for image in imageList:
        count += 1
        if(count % 500 == 0):
            print 'Image number:', count
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]
        #for each ground truth, see we have predicted a corresponding result
        for i in range(49):
            preds = out[i*25:i*25+25]
            row = i/7
            col = i%7
            box = image.boxes[row][col]
            if(box.has_obj):
                for obj in box.objs:
                    obj_num += 1
                    true_class = obj.class_num
                    #see what we predict
                    if(preds[24] > thresh):
                        predcit_class = np.argmax(preds[4:24])
                        if(predcit_class == true_class):
                            correct += 1
    return correct*1.0/obj_num

def MeasureAcc(model,sample_number,vocPath,imageNameFile):
    imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
    acc = Acc(imageList,model)
    re = Recall(imageList,model)

    return acc,re

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('Tiny_Yolo_weights_07_train.h5')

#Measure Test Accuracy
sample_number = 5011
vocPath= os.path.join(os.getcwd(),'dataset/train_val')
imageNameFile= os.path.join(vocPath,'ImageSets/Main/trainval.txt')

imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
acc = Acc(imageList,model)
re = Recall(imageList,model)

print "Accuracy and Recall are: ",acc,re
