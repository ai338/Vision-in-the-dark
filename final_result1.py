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



input_patch_list = np.load('input_patch_list.npy')
gt_patch_list = np.load('gt_patch_list.npy')
n = 15
filepath = "F:\\Learning-to-See-in-the-Dark\\checkpoints_1\\model-4750-0.69.hdf5"
model = models.load_model(filepath)
data = model.predict(np.reshape(input_patch_list[n], (1,512,512,9)))
data = np.minimum(np.maximum(data, 0), 1)
data = np.reshape(data, (1536,1536,3))
                
scipy.misc.toimage(data * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\final_result1\\' + 'img' + str(2) + '.png')


gt = gt_patch_list[n]
gt = np.minimum(np.maximum(gt, 0), 1)
gt = np.reshape(gt, (1536,1536,3))
scipy.misc.toimage(gt * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\final_result1\\' + 'gt' + '.png')

