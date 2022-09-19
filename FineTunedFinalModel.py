# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:20:19 2021

@author: shivu
"""

#Eshaan Vora
#EshaanVora@gmail.com

#import packages
#import tensorflow as tf
#from tensorflow import keras

import os, fnmatch, shutil
from keras.applications import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from keras.wrappers.scikit_learn import KerasClassifier

#TEST IF ENVIRONMENT IS CORRECTLY SET UP WITH NVIDIA GPU
#print(tf.test.is_built_with_cuda())

## RENAME TRAINING DATA
# RENAME PNEUMONIA PHOTOS
#file_path = 'chest_xrays/chest_xray/train/PNEUMONIA/'
#files_to_rename = fnmatch.filter(os.listdir(file_path), '*.jpeg')
#new_name = 'lung'
#for i, file_name in enumerate(files_to_rename):
    #new_file_name = new_name + str(i) + '.jpeg'
    #os.rename(file_path + file_name,
         # file_path + new_file_name)

# RENAME NORMAL PHOTOS
#file_path = 'chest_xrays/chest_xray/train/NORMAL/'
#files_to_rename = fnmatch.filter(os.listdir(file_path), '*.jpeg')
#new_name = 'normal'
#for i, file_name in enumerate(files_to_rename):
    #new_file_name = new_name + str(i) + '.jpeg'
    #os.rename(file_path + file_name,
         #file_path + new_file_name)


conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))

base_dir = 'pneumonia_and_normal_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

train_features, train_labels = extract_features(train_dir, 1340)
validation_features, validation_labels = extract_features(validation_dir, 675)
test_features, test_labels = extract_features(test_dir, 675)

train_features = np.reshape(train_features, (1340, 4*4* 512))
validation_features = np.reshape(validation_features, (675, 4*4* 512))
test_features = np.reshape(test_features, (675, 4*4* 512))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=33,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=16)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.ylim([0.5, 1])
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.ylim([0, 1])
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=16)
print('test acc:', test_acc)
print('test loss:', test_loss)

#print(test_generator)
#predictions = model.predict(test_generator)
#print(predictions)
#print(conv_base.summary())


######################################################################
#Smoothed Graphs
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
