# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:49:09 2021

@author: shivu
"""

import os, fnmatch, shutil

## CREATE NEW DIRECTORIES

original_dataset_dir = 'train'
base_dir = 'pneumonia_and_normal_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')
os.mkdir(train_pneumonia_dir)

train_normal_dir = os.path.join(train_dir, 'normal')
os.mkdir(train_normal_dir)

validation_pneumonia_dir = os.path.join(validation_dir, 'pneumonia')
os.mkdir(validation_pneumonia_dir)

validation_normal_dir = os.path.join(validation_dir, 'normal')
os.mkdir(validation_normal_dir)

test_pneumonia_dir = os.path.join(test_dir, 'pneumonia')
os.mkdir(test_pneumonia_dir)

test_normal_dir = os.path.join(test_dir, 'normal')
os.mkdir(test_normal_dir)

fnames = ['lung{}.jpeg'.format(i) for i in range(670)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_pneumonia_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['lung{}.jpeg'.format(i) for i in range(670, 1005)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_pneumonia_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['lung{}.jpeg'.format(i) for i in range(1005 , 1340)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_pneumonia_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['normal{}.jpeg'.format(i) for i in range(670)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_normal_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['normal{}.jpeg'.format(i) for i in range(670, 1005)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_normal_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['normal{}.jpeg'.format(i) for i in range(1005, 1340)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_normal_dir, fname)
    shutil.copyfile(src, dst)
