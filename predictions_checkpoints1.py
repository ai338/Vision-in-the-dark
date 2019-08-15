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

input_patch_list = np.load('input_patch_list.npy')

# ucitaj u foru model koji treba 
for i in range(9):
    filepath="F:\\Learning-to-See-in-the-Dark\\checkpoints_1\\model-{i:02d}.h5" # potencijalno stavi hdf5 
    model = models.load_model(filepath) 
    data = model.predict(np.reshape(input_patch_list[10], (1,512,512,9)))
    data = np.reshape(data, (1536,1536,3))
    img = Image.fromarray(data, 'RGB')
    img_name = 'F:\\Learning-to-See-in-the-Dark\\predictions_checkpoints1\\' + 'img' + str(i) + '.png'
    img.save(img_name)

    