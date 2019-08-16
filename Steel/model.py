import os
import json

import cv2
import keras

from keras import backend as k
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Preprocessing

# Importing the datasets

train_df = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/Steel/train.csv')

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[0])
train_df['Classid'] = train_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df.shape

mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
mask_count_df.shape

submission_df = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/Steel/sample_submission.csv')
submission_df['ImageId'] = submission_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[0])


def mask2rle(img):
    ''' img: numpy array, 1- mask, 0- background
    returns run length as string formatted
    '''

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]+1
    runs[1::2] -= runs[::2]
    return ''.join(str(x)for x in runs)


def rle2mask(mask_rle, shape=(256, 1600)):
    '''
    mask_rle : run length as string formatted (start length)
    shape: length and width of array to return
    returns numpy array, 1-mask, 0-background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts+length
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

    def build_masks(rles, input_shape):
        depth = len(rles)
        height, width = input_shape
        masks = np.zeros((height, width, depth))

        for i, rle in enumerate(rles):
            if type(rle) is str:
                masks[:, :, i] = rle2mask(rle, (width, height))

        return masks

    def build_rles(masks):
        width, height, depth = shape(masks)
        rles = [mask2rle(masks[:, :, i]) for i in range(depth)]

        return rles
