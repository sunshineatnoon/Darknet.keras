import numpy as np
import theano.tensor as T
import theano
import os
from PIL import Image
from PIL import ImageDraw


y1,y2 = T.dmatrices('y1', 'y2')
loss = 0.0

scale_vector = []
scale_vector.extend([2]*4)
scale_vector.extend([1]*20)
scale_vector = np.reshape(np.asarray(scale_vector),(1,len(scale_vector)))

for i in range(2):
    y1_piece = y1[:,i*25:i*25+24]
    y2_piece = y2[:,i*25:i*25+24]

    y1_piece = y1_piece * scale_vector
    y2_piece = y2_piece * scale_vector

    loss_piece = T.sum(T.square(y1_piece - y2_piece),axis=1)
    loss = loss + loss_piece * y2[:,i*25+24]
    
    closs = T.square(y2[:,i*25+24] - y1[:,i*25+24])
    cmask = (1 - y2[:,i*25+24])*0.5 + y2[:,i*25+24]
    closs = closs * cmask
    loss = loss + closs

loss = T.sum(loss)

f = theano.function([y1,y2], [loss])

y_pred = np.asarray([0.5,0.5,0.1,0.2,0.1,0.2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6,0.5,0.5,0.1,0.2,0.1,0.2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6])
y_true = np.asarray([0.5,0.1,0.2,0.4,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_pred = np.expand_dims(y_pred,axis=0)
y_true = np.expand_dims(y_true,axis=0)
print f(y_pred,y_true)

'''
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
imgPath = os.path.join(os.getcwd(),'images/bike_resize.jpg')
img = Image.open(imgPath)
drawable = ImageDraw.Draw(img)

preds = [0.16017964,0.815,0.89554453,0.79857658,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
thresh = 0.1

if(preds[24] > thresh):

    row = 3
    col = 4
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
    img.save('dog_detection.jpg')
'''

'''
y1,y2 = T.matrices('y1','y2')
loss = 0.0
for i in range(2):
    y1_piece = y1[:,i*5:i*5+4]
    y2_piece = y2[:,i*5:i*5+4]
    loss_piece = T.sum(T.square(y1_piece - y2_piece),axis=1)
    loss = loss + loss_piece * y2[:,i*5+4]
    loss = loss + T.square(y2[:,i*5+4] - y1[:,i*5+4])
loss = T.sum(loss)

f = theano.function([y1,y2],[loss]) #y1:y_pred, y2:y_true

print f(np.asarray([[1,2,8,7,1,2,3,8,4,1],[3,2,1,5,0,9,1,8,3,1]]),np.asarray([[1,3,1,6,1,4,4,1,5,0],[2,8,0,5,0,1,2,3,9,1]]))
'''
