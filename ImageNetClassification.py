# This is all 'experimental code'; it loosely follows the instructions from 
# the 'ImageNet classification' and 'Filter visualization' ipynb tutorials from 
# http://caffe.berkeleyvision.org/

import cPickle
import gzip
import os
import sys
import time

import scipy
import scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../../caffe-master/'
# this file is expected to be in {caffe_root}/examples

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Make sure that this first links to the 'caffe-master' directory 
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../../caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../../caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/cat.jpg'

import os
#os.system("echo 'hello world'")
if not os.path.isfile(PRETRAINED):
    print("Downloading pre-trained CaffeNet model...")
    os.system("../../caffe-master/scripts/download_model_binary.py ../../caffe-master/models/bvlc_reference_caffenet")

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
	                       channel_swap=(2,1,0),
	                       raw_scale=255,
	                       image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

#plt.show()

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
#print(prediction[0])
#plt.show()


prediction = net.predict([input_image], oversample=False)
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()

#plt.show()

print(net.image_dims)

resized_image = caffe.io.resize_image(input_image, net.image_dims)
#plt.imshow(resized_image)
#plt.show()

#%timeit net.predict([input_image])

# Resize the image to the standard (256, 256) and oversample net input sized crops.
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.


# now try and extract output from 2nd to last layer
# starting filter visualization tutorial 

print '... hello world'


