from __future__ import division
import os, time, scipy.io
#import tensorflow as tf
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

from sklearn.utils import shuffle

input_dir = './dataset/Fuji/short/'
gt_dir = './dataset/Fuji/long/'
checkpoint_dir = './result_Fuji/'
result_dir = './result_Fuji/'


# get train IDs
train_fns = glob.glob(gt_dir + '0*.RAF')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
ps = 512  # patch size for training


def pack_raw(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out


gt_images = [None] * 6000
in_images = {}
in_images['300'] = [None] * len(train_ids)
in_images['250'] = [None] * len(train_ids)
in_images['100'] = [None] * len(train_ids)

input_patch_list = []
gt_patch_list = []
for iteration in range(0, 3):
    # if os.path.isdir(result_dir + '%04d' % epoch):
    #     continue
    # cnt = 0
    # if epoch > 2000:
    #     learning_rate = 1e-5
    print('iteration no:' + str(iteration))
    i = 0 
    for ind in np.random.permutation(len(train_ids)):
        i = i+1
        print('picture no' + str(iteration) + '.' + str(i))
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.RAF' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

      

        if in_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            in_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = in_images[str(ratio)[0:3]][ind].shape[1]
        W = in_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = in_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 3:yy * 3 + ps * 3, xx * 3:xx * 3 + ps * 3, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        input_patch_list.append(input_patch)
        gt_patch_list.append(gt_patch)
        

print('done preprocessing')
input_patch_list = np.asarray(input_patch_list)
gt_patch_list = np.asarray(gt_patch_list)

input_patch_list = input_patch_list[:,0,:,:,:]
gt_patch_list = gt_patch_list[:,0,:,:,:]

np.save('input_patch_list.npy', input_patch_list)
print('done saving')
np.save('gt_patch_list.npy', gt_patch_list)
