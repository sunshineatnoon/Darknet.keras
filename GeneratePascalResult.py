from keras.models import model_from_json
import theano.tensor as T

from utils.readImgFile import readImg
from utils.crop import crop_detection
from utils.ReadPascalVoc2 import prepareBatch

import os
import numpy as np
from PIL import Image
from PIL import ImageDraw

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
    fcat = open('comp4_det_test_cat.txt','write')
    fcar = open('comp4_det_test_car.txt','write')
    fbottle = open('comp4_det_test_bottle.txt','write')
    ftrain = open('comp4_det_test_train.txt','write')


    count = 0
    for image in imageList:
        img = Image.open(image.imgPath)
        img_name = image.imgPath.split('/')[-1].split('.')[0]
        W,H = img.size
        drawable = ImageDraw.Draw(img)

    	count += 1
        if(count % 500 == 0):
            print 'Image number:', count

        #Get prediction from neural network
        imgg = crop_detection(image.imgPath,new_width=448,new_height=448)
        imgg = np.expand_dims(imgg, axis=0)
        out = model.predict(imgg)
        predictions = out[0]

    	#Post process predicting results
    	boxes = convert_yolo_detections(predictions)
    	boxes = do_nms_sort(boxes)

        for i in range(len(boxes)):
            box = boxes[i]
            prob = np.max(box.probs)
            class_num = boxes[i].class_num
            if((class_num in [4,6,7,18]) and prob >= 0.2):#if this box contains a car

                centerx = boxes[i].x
                centery = boxes[i].y
                h = boxes[i].h
                w = boxes[i].w

                left = centerx - w/2.0
                right = centerx + w/2.0
                up = centery - h/2.0
                down = centery + h/2.0

                if(left < 0): left = 0
                if(right > 448): right = 447
                if(up < 0): up = 0
                if(down > 448): down = 447

                #convert the coords to image before streching to 448*448
                left = left/448.0 * W
                right = right/448.0 * W
                up = up/448.0 * H
                down = down/448.0 * H

                #drawable.rectangle([left,up,right,down],outline="red")
                if(class_num == 4):
                    fbottle.write(img_name+' '+str(round(prob,6))+' '+str(round(left,6))+' '+str(round(up,6))+' '+str(round(right,6))+' '+str(round(down,6))+'\n')
                elif(class_num == 6):
                    fcar.write(img_name+' '+str(round(prob,6))+' '+str(round(left,6))+' '+str(round(up,6))+' '+str(round(right,6))+' '+str(round(down,6))+'\n')
                elif(class_num == 7):
                    fcat.write(img_name+' '+str(round(prob,6))+' '+str(round(left,6))+' '+str(round(up,6))+' '+str(round(right,6))+' '+str(round(down,6))+'\n')
                elif(class_num == 18):
                    ftrain.write(img_name+' '+str(round(prob,6))+' '+str(round(left,6))+' '+str(round(up,6))+' '+str(round(right,6))+' '+str(round(down,6))+'\n')
        #img.save(os.path.join(os.getcwd(),'results',img_name+'.jpg'))

    return

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('Tiny_Yolo_weights_07_12_33102_iter2.h5')

#Measure Test Accuracy
sample_number = 4952
vocPath= os.path.join(os.getcwd(),'dataset/VOCdevkit/VOC2007')
imageNameFile= os.path.join(vocPath,'ImageSets/Main/test.txt')

imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
Acc(imageList,model)
