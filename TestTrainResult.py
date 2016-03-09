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

model = model_from_json(open('Tiny_Yolo_Architecture.json').read(),custom_objects={'custom_loss':custom_loss})
model.load_weights('weights2.hdf5')

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
	img,im = crop_detection(os.path.join(os.getcwd(),img_name),new_width=448,new_height=448,save=True)
	im.save(os.getcwd()+'/images/'+img_name.split('/')[1].split('.')[0]+'_resize.jpg')
	img = np.expand_dims(img, axis=0)
	out = model.predict(img)
	out = out[0]

	#Post process predicting results
	thresh = 0.3
	imgPath = os.path.join(os.getcwd(),img_name)
	img = Image.open(imgPath)
	img_draw = Image.open(os.getcwd()+'/images/'+img_name.split('/')[1].split('.')[0]+'_resize.jpg')
	drawable = ImageDraw.Draw(img_draw)
	labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
	for i in range(49):
		preds = out[i*25:(i+1)*25]
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

			drawable.rectangle([left,up,right,down],outline='red')
			print 'Class is: ',labels[np.argmax(preds[4:24])]
			print np.max(preds[4:24])
	print
	print
	img_draw.save(os.path.join(os.getcwd(),'results',img_name.split('/')[1]))
