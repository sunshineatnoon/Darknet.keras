import os
path = os.path.join('/home/media/Documents/YOLO.keras/dataset/VOC2012/JPEGImages')
f = open('07_12_train.txt','w')
for root,dirs,files in os.walk(path):
	for fn in files:
		f.write(fn.split('.')[0] + '\n')
f.close()
