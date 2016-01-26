# Darknet.keras
This is a transfer from weights trained by [Darknet](http://pjreddie.com/darknet/) to [keras](http://keras.io/) with Theano backend.

Currently I only finish test process for [YOLO Tiny Model](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-tiny.cfg).

Due to image preprocess difference bewteen my reimplementaion and Darknet's, the result has slightly difference.

To run the code:

1. Create a folder called weights
2. Download yolo-tiny.weights from [Darknet website](http://pjreddie.com/darknet/yolo/)
3. Run:
   ```
   python RunTinyYolo.py
   ```
The detection result will be saved in the results folder

Notes:
Make sure your have Theano and Keras installed
