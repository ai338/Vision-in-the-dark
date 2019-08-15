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
#from keras.models import Model 
#from keras.models import load_model


input_dir = './dataset/Fuji/short/'
gt_dir = './dataset/Fuji/long/'
checkpoint_dir = './result_Fuji/'
result_dir = './result_Fuji/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.RAF')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
callbacks_list = [tensorboard_callback]




def log_image(file_writer, tensor, epoch_no, tag):
    height, width, channel = tensor.shape
    tensor = ((tensor + 1) * 255)
    tensor = tensor.astype('uint8')
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    tf_img = Summary.Image(height=height,
                              width=width,
                              colorspace=channel,
                              encoded_image_string=image_string)
    summary = Summary(value=[Summary.Value(tag=tag, image=tf_img)])
    file_writer.add_summary(summary, epoch_no)
    file_writer.flush()
def dts(x):
    import tensorflow as tf 
    return tf.depth_to_space(x,3)
#def lrelu(x):
 #   return tf.maximum(x * 0.2, x)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'random_uniform', padding = 'same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # second layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def get_unet(n_filters = 16, dropout = 0.1, batchnorm = False):
    # nismo sigurne da li treba raditi batch norm 
    # potencijalno n_filters = 32

    # Contracting Path
    input_layer = layers.Input(shape=(512,512,9))
    c1 = conv2d_block(input_layer, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    #p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    #p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    #p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path #pool size je u radu 2, potencijalno je greska = kernel size==pool size 
    u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    #u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    #u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    #u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1])
    #u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c9 = layers.Conv2D(27, (1, 1), activation=None)(c9)
    #outputs = tf.depth_to_space(c9, 3)
    
    outputs = layers.Lambda(dts)(c9)
    model = models.Model(inputs=input_layer, outputs=outputs)
    return model


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

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0

# for folder in allfolders:
#     lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
input_patch_list = []
gt_patch_list = []
# # for iteration in range(0, 4):
#     # if os.path.isdir(result_dir + '%04d' % epoch):
#     #     continue
#     # cnt = 0
#     # if epoch > 2000:
#     #     learning_rate = 1e-5
#     print(iteration)

#     for ind in np.random.permutation(len(train_ids)-132):
#         start=time.time()
#         print(len(train_ids))
#         print(start)
#         # get the path from image id
#         train_id = train_ids[ind]
#         in_files = glob.glob(input_dir + '%05d_00*.RAF' % train_id)
#         in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
#         in_fn = os.path.basename(in_path)
        
#         gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % train_id)
#         gt_path = gt_files[0]
#         gt_fn = os.path.basename(gt_path)
#         in_exposure = float(in_fn[9:-5])
#         gt_exposure = float(gt_fn[9:-5])
#         ratio = min(gt_exposure / in_exposure, 300)
#         print(ratio)
#         # st = time.time()
#         # cnt += 1

#         if in_images[str(ratio)[0:3]][ind] is None:
#             raw = rawpy.imread(in_path)
#             print('pack_raw begins')
#             in_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
             
#             print('pack_raw done')
#             gt_raw = rawpy.imread(gt_path)
#             im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#             gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

#         # crop
#         H = in_images[str(ratio)[0:3]][ind].shape[1]
#         W = in_images[str(ratio)[0:3]][ind].shape[2]
#         print('crop')
#         xx = np.random.randint(0, W - ps)
#         yy = np.random.randint(0, H - ps)
#         input_patch = in_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
#         gt_patch = gt_images[ind][:, yy * 3:yy * 3 + ps * 3, xx * 3:xx * 3 + ps * 3, :]
        
#         # if np.random.randint(2, size=1)[0] == 1:  # random flip
#         #     input_patch = np.flip(input_patch, axis=1)
#         #     gt_patch = np.flip(gt_patch, axis=1)
#         # if np.random.randint(2, size=1)[0] == 1:
#         #     input_patch = np.flip(input_patch, axis=2)
#         #     gt_patch = np.flip(gt_patch, axis=2)
#         # if np.random.randint(2, size=1)[0] == 1:  # random transpose
#         #     input_patch = np.transpose(input_patch, (0, 2, 1, 3))
#         #     gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

#         input_patch = np.minimum(input_patch, 1.0)
#         input_patch_list.append(input_patch)

#         gt_patch_list.append(gt_patch)
#         print(len(input_patch_list))
#         end=time.time()
#         print(end-start)

input_patch_list = np.asarray(input_patch_list)
gt_patch_list = np.asarray(gt_patch_list)

input_patch_list = input_patch_list[:,0,:,:,:]
gt_patch_list = gt_patch_list[:,0,:,:,:]

model_net = get_unet(16, dropout=0.1, batchnorm=False)
model_net.compile(optimizer='adam', loss = 'mean_absolute_error',metrics=['accuracy']) 
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period = 100) #treba promeniti pre pokretanja!!! 
callbacks_list.append(checkpoint)


input_patch_list, gt_patch_list = shuffle(input_patch_list, gt_patch_list)

m = input_patch_list.shape[0]

input_patch_list_train, gt_patch_list_train = input_patch_list[:int(0.8*m)], gt_patch_list[:int(0.8*m)]
input_patch_list_val, gt_patch_list_val = input_patch_list[int(0.8*m):], gt_patch_list[int(0.8*m):]


# for i in range(10000):
#     history = model_net.fit(input_patch_list_train, gt_patch_list_train, validation_data = [input_patch_list_val, gt_patch_list_val], epochs = 1, batch_size = 32, callbacks=callbacks_list)
#     data = model_net.predict(np.reshape(input_patch_list[10], (1,512,512,9)))
#     #np.save("img{}".format(i),img)
#     data = np.reshape(data, (1536,1536,3))
#     img = Image.fromarray(data, 'RGB')
#     #img.save("img{}".format(i))
#     img_name = 'F:\Learning-to-See-in-the-Dark\predictions1\\' + 'img' + str(i) + '.png'
#     img.save(img_name)
#     #img.show()


print('\nhistory dict:', history.history)

#model_net.save('my_model.h5')
#models.Model.save(model_net, 'my_model.h5')

models.save_model(model_net, 'my_model.h5')

#modelX = models.load_model('my_model.h5')
#modelX.summary()

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.show()


#ROOT_LOGS = 'F:\Learning-to-See-in-the-Dark\logs'
#callbacks = TensorBoard(ROOT_LOGS + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#file_writer = tensorboard_callback.writer
#log_image(file_writer, image, epoch_number) #image = generisana slika
#cnt = 0
#for img in gt_patch_list:
#    log_image(file_writer, img, cnt, tag = 'GT')
print('end')
#print(modelX.predict(input_patch_list, batch_size=1))