from keras.models import model_from_json
import theano.tensor as T

from utils.readImgFile import readImg
from utils.crop import crop_detection
from utils.ReadPascalVoc2 import prepareBatch

import os
import numpy as np

class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.row = 0
        self.col = 0
        self.class_num = 0
        self.probs = np.zeros((classes,1))

def convert_yolo_detections(predictions,classes=20,side=7,thresh=0.3):
    boxes = []
    for i in range(side*side):
        preds = predictions[i*25:(i+1)*25]
        if(preds[24] > thresh):
            #print preds[0:4],preds[24]
            row = i/7
            col = i%7
            #print row,col
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

            new_box = box(classes)
            new_box.x = centerx
            new_box.y = centery
            new_box.h = down-up
            new_box.w = right-left
            new_box.row = row
            new_box.col = col

            #normalize the class probabilities so they added to 1
            prob = preds[4:4+classes]
            new_box.probs = prob
            new_box.class_num = np.argmax(new_box.probs)
            boxes.append(new_box)
    return boxes

def prob_compare(boxa,boxb):
    if(boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]):
        return 1
    elif(boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]):
        return 0
    else:
        return -1

def do_nms_sort(boxes,classes=20,thresh=0.5):
	total = len(boxes)
	for box in boxes:
		box.class_num = np.argmax(box.probs)

	for i in range(len(boxes)):
		for j in range(i+1,total):
			boxa = boxes[i]
			boxb = boxes[j]
			iou = box_iou(boxa,boxb)
			if(iou > thresh):
				if(boxa.class_num == boxb.class_num):
					if(boxa.probs[boxa.class_num] > boxb.probs[boxb.class_num]):
						#get rid of boxb
						boxb.probs -= boxb.probs #set all probs to 0
						boxes[j] = boxb
					else:
						#get rid of boxa
						boxa.probs -= boxa.probs
						boxes[i] = boxa
	return boxes

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

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
        predictions = out[0]

    	#Post process predicting results
    	boxes = convert_yolo_detections(predictions)
    	boxes = do_nms_sort(boxes)

        for i in range(len(boxes)):
            preds = boxes[i].probs
            not_all_zeros = np.any(preds)
            if(not_all_zeros):
                object_num += 1
                x = boxes[i].x
                y = boxes[i].y
                h = boxes[i].h
                w = boxes[i].w
                class_num = boxes[i].class_num
                row = boxes[i].row
                col = boxes[i].col
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
model.load_weights('weights2.hdf5')

#Measure Test Accuracy
sample_number = 4952
vocPath= os.path.join(os.getcwd(),'dataset/VOCdevkit/VOC2007')
imageNameFile= os.path.join(vocPath,'ImageSets/Main/test.txt')

imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
acc = Acc(imageList,model)
#re = Recall(imageList,model)

#print "Accuracy and Recall are: ",acc,re
print "Accuracy is: ",acc
