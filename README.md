# Darknet.keras
This is a transfer from weights trained by [Darknet](http://pjreddie.com/darknet/) to [keras](http://keras.io/) with Theano backend.

Currently I only finish test process for [YOLO Tiny Model](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-tiny.cfg).

Due to image preprocess difference bewteen my reimplementaion and Darknet's, the result has slightly difference.

To run the code:

1. Create three folders:  `weights, results, images`.
2. Put the images you want to detect in the folder `images`.
2. Download yolo-tiny.weights from [Darknet website](http://pjreddie.com/darknet/yolo/) and put it in the folder `weights`.
3. Run:
   ```
   python RunTinyYolo.py
   ```
The detection result will be saved in the results folder

Notes:
Make sure your have Theano and Keras installed

**IMPORTANT WARNING**

**This code is written for older version of YOLO weight, now the weight has updated to version 2.0, so the code is no longer working. In order to make it work, one has to change [ReadTinyYOLONetWeights](https://github.com/sunshineatnoon/Darknet.keras/blob/master/utils/TinyYoloNet.py#L55) to read new YOLO weight bite by bite.**
