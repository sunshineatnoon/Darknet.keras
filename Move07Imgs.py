import os
from shutil import copy

"""
This script moves all training data of VOC07 to the folder of VOC12, so  that I
train the model on train samples of both VOC07 and VOC12.
"""
path07 = os.path.join(os.getcwd(),'dataset/train_val')
path12 = os.path.join(os.getcwd(),'dataset/VOC2012')

f = open(os.path.join(os.getcwd(),'dataset/train_val/imageNames.txt'))
lines = f.readlines()
for line in lines:
    name = line.strip('\n')

    ImgSrc = os.path.join(path07,'JPEGImages',name+'.jpg')
    AnnoSrc = os.path.join(path07,'Annotations',name+'.xml')

    ImgDst = os.path.join(path12,'JPEGImages/')
    AnnoDst = os.path.join(path12,'Annotations/')

    copy(ImgSrc,ImgDst)
    copy(AnnoSrc,AnnoDst)
