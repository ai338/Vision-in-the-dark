from __future__ import division
import os, time, scipy.io
from tensorflow import keras
from keras import layers
from keras import models
import numpy as np
import rawpy
import glob
import datetime
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow import Summary
from tensorflow import summary
from PIL import Image
from tensorflow import device


print('pocinjemooooooo')
test_list = np.load('test_list.npy')
gt_list = np.load('gt_list.npy')
print(test_list[0].shape)

i = 1
n = 0
for root, dirs, files in os.walk("F:\\Learning-to-See-in-the-Dark\\checkpoints_1", topdown=False):
    for name in files:
        if "model" in name:
            filepath = os.path.join(root, name)
            with device('/cpu:0'):
                model = models.load_model(filepath)
                
                data = model.predict(np.reshape(test_list[n],  (1,512,512,9)))
              
                data = np.minimum(np.maximum(data, 0), 1)
                data = np.reshape(data, (1536,1536,3))

                
                scipy.misc.toimage(data * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\predictions1\\' + 'img' + str(i) + '.png')
                        
                i = i + 1

                # img = Image.fromarray(data, 'RGB')
                # img_name = 'F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'img' + str(i) + '.png'
                # i = i + 1
                # img.save(img_name)


gt = gt_list[n]
gt = np.minimum(np.maximum(gt, 0), 1)
gt = np.reshape(gt, (1536,1536,3))
scipy.misc.toimage(gt * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\predictions1\\' + 'gt' + '.png')



# output = data
# _, H, W, _ = output.shape
# output = output[0, :, :, :]
# scale_full = scale_full[0, 0:H, 0:W, :]
# scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)