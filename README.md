# Darknet.keras
This is a transfer from weights trained by [Darknet](http://pjreddie.com/darknet/) to [keras](http://keras.io/) with Theano backend.

Currently I only transfered weigths of the [extraction net](https://github.com/pjreddie/darknet/blob/master/cfg/extraction.cfg).

Due to image preprocess difference bewteen my reimplementaion and Darknet's, the result probability is different, but predicted labels are same.

To run the code:

1. Create a folder called weights
2. Download extraction.weights from [Darknet website](http://pjreddie.com/darknet/imagenet/)
3. Run:
   ```
   python Run.py
   ```

You should see output like this:

```
malamute : 0.654673
Eskimo dog : 0.157867
Siberian husky : 0.073547
dogsled : 0.029347
Border collie : 0.014436
Tibetan mastiff : 0.007743
Cardigan : 0.007715
Pembroke : 0.006594
German shepherd : 0.005453
bicycle-built-for-two : 0.003298
```

If you would like to see exact result of the Darknet Extraction model, change line 50 and 51 to this:

```
image = readImg(os.path.join(os.getcwd(),'images/dog.file'))
#image = crop(os.path.join(os.getcwd(),'images/dog.jpg'))
```
And then run python `Run.py`, you shoule see something like this:

```
malamute : 0.580025
Eskimo dog : 0.232404
Siberian husky : 0.073097
dogsled : 0.046569
Border collie : 0.007437
Tibetan mastiff : 0.006056
German shepherd : 0.005579
bicycle-built-for-two : 0.004949
Pembroke : 0.004684
Cardigan : 0.003681
```
Notes:
Make sure your have Theano and Keras installed
