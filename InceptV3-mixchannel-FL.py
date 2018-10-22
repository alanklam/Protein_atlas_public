# This is a Python script to apply transfer learning for the Human Protein Atlas Classification competition
# Public leader board rank 26th (F1 score 0.433)
#Author: Kin Lam (Oct 2018)

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle

import warnings
warnings.filterwarnings("ignore")
SIZE = 299

# Load dataset info
path_to_train = 'data/train/'
data = pd.read_csv('data/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

# define image data generator
## 1. cell image has 4 channels (RGBY), put target protein channel at green, 
## combine R+Y and B+Y to the remaining two
## 2. resize 512x512 image to 299x299 to use pretrained model
## 3. data augmentation by one of (rotate 0,90,180,270 degree, flipping left-right, flipping up-down)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)   
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
        (np.array(image_red_ch)+np.array(image_yellow_ch))/2, 
        np.array(image_green_ch), 
        (np.array(image_blue_ch)+np.array(image_yellow_ch))/2), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model
import tensorflow as tf

# define model, load InceptionV3 net from Keras, 
# connect to a 1x1 conv2D layer and then to dense 1024, finally to a 28-node classifers    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 100; batch_size = 16
checkpoint = ModelCheckpoint('models/InceptV3-4C-focalloss2.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                                   verbose=1, min_lr=1e-6, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)

# create focal loss function
def focal_loss(gamma=2., alpha=.25):
    #default paramemters in Lin et al. (2017)
    def focal_loss_eval(y_true, y_pred):
         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
         _epsilon = tf.convert_to_tensor(1e-7,dtype=pt_1.dtype.base_dtype)
         pt_1 = tf.clip_by_value(pt_1,_epsilon,1-_epsilon)
         pt_0 = tf.clip_by_value(pt_0,_epsilon,1-_epsilon)
         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_eval

# evaluate model by marco-averaging F1 score
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred,'float'),axis=-1)
    fp = K.sum(K.cast((1-y_true)*y_pred,'float'),axis=-1)
    fn = K.sum(K.cast(y_true*(1-y_pred),'float'),axis=-1)
    
    p = tp/(tp+fp+K.epsilon())
    r = tp/(tp+fn+K.epsilon())
    
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1),tf.zeros_like(f1),f1)
    return K.mean(f1)

callbacks_list = [checkpoint, early, reduceLROnPlat]

# split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

# create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)

# warm up model
model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=28)

for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

model.compile(
    loss=[focal_loss(gamma=2.0,alpha=0.5)], 
    optimizer=Adam(1e-03),
    metrics=[f1])
#%%
# model.summary()
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=2, 
    verbose=2)

# train all layers 
for layer in model.layers:
    layer.trainable = True
model.compile(loss=[focal_loss(gamma=2.0,alpha=0.5)],
            optimizer=Adam(1e-04),
            metrics=[f1])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs, 
    verbose=2,
    callbacks=callbacks_list)
np.save('prediction/loss_InceptV3-4C-focalloss2.npy',history.history["loss"])
np.save('prediction/valloss_InceptV3-4C-focalloss2.npy',history.history["val_loss"])

#%%    
# Create submission
submit = pd.read_csv('data/sample_submission.csv')
predicted = []
draw_predict = []
model.load_weights('models/InceptV3-4C-focalloss.h5')
#%%
for name in tqdm(submit['Id']):
    path = os.path.join('data/test/', name)
    image = data_generator.load_image(path, (SIZE,SIZE,3))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)
    label_predict = np.arange(28)[score_predict>=0.45]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
np.save('prediction/draw_predict_InceptV3-4C-focalloss.npy', score_predict)
submit.to_csv('submission/InceptV3-4C-focalloss-p45.csv', index=False)

