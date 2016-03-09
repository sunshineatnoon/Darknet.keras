from keras.models import model_from_json
import theano.tensor as T
from utils.readImgFile import readImg
from utils.crop import crop_detection
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw

def custom_loss(y_true,y_pred):
	scale_vector = []
	scale_vector.extend([2]*4)
	scale_vector.extend([0]*20)
	scale_vector = scale_vector * 49

	mask = y_true > 0
	y_pred = y_pred * mask
	y_pred = y_pred * scale_vector
	y_true = y_true * scale_vector

	loss = T.mean(T.square(y_pred - y_true), axis=-1)
	return loss

class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
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

			#normalize the class probabilities so they added to 0
			prob = preds[4:4+classes]
			new_box.probs = prob
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

def draw_detections(impath,thresh,boxes,classes,labels,save_name):
	"""
	Args:
	    impath: The image path
	    num: total number of bounding boxes
	    thresh: boxes prob beyond this thresh will be drawn
	    boxes: boxes predicted by the network
	    classes: class numbers of the objects
	"""
	num = len(boxes)
	img = Image.open(impath)
	drawable = ImageDraw.Draw(img)
	W,H = img.size
	for i in range(num):
		#for each box, find the class with maximum prob
		max_class = np.argmax(boxes[i].probs)
		prob = boxes[i].probs[max_class]
		if(prob > thresh):
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

			print left,right,up,down
			drawable.rectangle([left,up,right,down],outline="red")
			img.save(os.path.join(os.getcwd(),'results',save_name))
			print labels[max_class],": ",boxes[i].probs[max_class]

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('Tiny_Yolo_weights_07_12_26720_iter2.h5')
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

#Predict output
#image = readImg(os.path.join(os.getcwd(),'Yolo_dog.img'),h=448,w=448)
images_path = '/home/media/Documents/YOLO.keras/images'
img_names = []
for root, dirs, files in os.walk(images_path):
	for name in files:
		img_names.append('images/'+name)

#img_names = ['images/eagle.jpg','images/000047.jpg','images/000009.jpg']
for img_name in img_names:
	print img_name
	img = crop_detection(os.path.join(os.getcwd(),img_name),new_width=448,new_height=448)
	img = np.expand_dims(img, axis=0)
	out = model.predict(img)
	predictions = out[0]

	#Post process predicting results
	boxes = convert_yolo_detections(predictions)
	boxes = do_nms_sort(boxes)

	#print len(boxes)
	draw_detections(os.path.join('/home/media/Documents/YOLO.keras/',img_name),0.2,boxes,20,labels,img_name.split('/')[1])
	print
	print
