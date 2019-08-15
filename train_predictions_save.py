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

# ucitaj u foru model koji treba
# for i in range(9):
#     filepath="F:\\Learning-to-See-in-the-Dark\\weights-improvement-01-0.87.hdf5" # potencijalno stavi hdf5
#     model = models.load_model(filepath)
#     data = model.predict(np.reshape(input_patch_list[10], (1,512,512,9)))
#     data = np.reshape(data, (1536,1536,3))
#     img = Image.fromarray(data, 'RGB')
#     img_name = 'F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'img' + str(i) + '.png'
#     img.save(img_name)



i = 1
n = 4
for root, dirs, files in os.walk("F:\\Learning-to-See-in-the-Dark\\checkpoints_1", topdown=False):
    for name in files:
        if "model" in name:
            filepath = os.path.join(root, name)
            with device('/cpu:0'):
                model = models.load_model(filepath)
                data = model.predict(np.reshape(input_patch_list[n], (1,512,512,9)))

                data = np.minimum(np.maximum(data, 0), 1)
                data = np.reshape(data, (1536,1536,3))
                
                scipy.misc.toimage(data * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'img' + str(i) + '.png')

                print(i)     
                i = i + 1

                # img = Image.fromarray(data, 'RGB')
                # img_name = 'F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'img' + str(i) + '.png'
                # i = i + 1
                # img.save(img_name)


gt = gt_patch_list[n]
gt = np.minimum(np.maximum(gt, 0), 1)
gt = np.reshape(gt, (1536,1536,3))
scipy.misc.toimage(gt * 255, high=255, low=0, cmin=0, cmax=255).save('F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'gt' + '.png')



# output = data
# _, H, W, _ = output.shape
# output = output[0, :, :, :]
# scale_full = scale_full[0, 0:H, 0:W, :]
# scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)