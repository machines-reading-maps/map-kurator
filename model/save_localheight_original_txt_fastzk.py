# -*- coding: utf-8 -*-
import glob
import json

import cv2
import math
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import tensorflow as tf

print(tf.__file__)
print(tf.__version__)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


os.environ['CUDA_VISIBLE_DEVICES'] = ""
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, AveragePooling2D, Reshape
from keras.layers import UpSampling2D, ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.layers import Lambda, TimeDistributed
from keras import layers

import numpy as np
import cv2
import argparse
import glob


from loss import weighted_categorical_crossentropy, mean_squared_error_mask
from loss import mean_absolute_error_mask, mean_absolute_percentage_error_mask
from mymodel import model_U_VGG_Centerline_Localheight


map_images = glob.glob('./data/test_imgs/sample_input/*.png')

output_path = './data/test_imgs/sample_output/'

saved_weights = './data/l_weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5'
model = model_U_VGG_Centerline_Localheight()
model.load_weights(saved_weights)

if not os.path.isdir(output_path):
    os.makedirs(output_path)


shift_size = 512

for map_path in map_images:

    base_name = os.path.basename(map_path)

    txt_name = output_path + base_name[0:len(base_name) - 4] + '.txt'

    f = open(txt_name, 'w+')

    print(map_path)

    map_img = cv2.imread(map_path)

    width = map_img.shape[1]  # dimension2
    height = map_img.shape[0]  # dimension1

    in_map_img = map_img / 255.
    
    # pad the image to the size divisible by shift-size
    num_tiles_w = int(np.ceil(1. * width/shift_size))
    num_tiles_h = int(np.ceil(1. * height/shift_size))
    enlarged_width = int(shift_size * num_tiles_w)
    enlarged_height = int(shift_size * num_tiles_h)
    print("BLAGALHAGLAHGA:")
    print(f"{width}-{num_tiles_w}, {height}-{num_tiles_h}, {enlarged_width}, {enlarged_height}")
    # paste the original map to the enlarged map
    enlarged_map = np.zeros((enlarged_height, enlarged_width, 3)).astype(np.float32)
    enlarged_map[0:height, 0:width, :] = in_map_img
    
    # define the output probability maps
    localheight_map_o = np.zeros((enlarged_height, enlarged_width, 1), np.float32)
    center_map_o = np.zeros((enlarged_height, enlarged_width, 2), np.float32)
    prob_map_o = np.zeros((enlarged_height, enlarged_width, 3), np.float32)
    
    # process tile by tile
    for idx in range(0, num_tiles_h):
        # pack several tiles in a batch and feed the batch to the model
        test_batch = []
        for jdx in range(0, num_tiles_w):
            img_clip = enlarged_map[idx*shift_size:(idx+1)*shift_size, jdx*shift_size:(jdx+1)*shift_size, :]
            test_batch.append(img_clip)
        test_batch = np.array(test_batch).astype(np.float32)
        
        # use the pretrained model to predict
        batch_out = model.predict(test_batch)
        
        # get predictions
        prob_map_batch = batch_out[0]
        center_map_batch = batch_out[1]
        localheight_map_batch = batch_out[2]
        
        # paste the predicted probabilty maps to the output image
        for jdx in range(0, num_tiles_w):
            localheight_map_o[idx*shift_size:(idx+1)*shift_size, jdx*shift_size:(jdx+1)*shift_size, :] = localheight_map_batch[jdx]
            center_map_o[idx*shift_size:(idx+1)*shift_size, jdx*shift_size:(jdx+1)*shift_size, :] = center_map_batch[jdx]
            prob_map_o[idx*shift_size:(idx+1)*shift_size, jdx*shift_size:(jdx+1)*shift_size, :] = prob_map_batch[jdx]
    
    
    # convert from 0-1? to 0-255 range
    prob_map_o = (prob_map_o * 255).astype(np.uint8)
    center_map_o = (center_map_o[:, :, 1] * 255).astype(np.uint8)
    #localheight_map = (localheight_map_o * 255).astype(np.uint8)
    
    prob_map_o = prob_map_o[0:height, 0:width, :]
    center_map_o = center_map_o[0:height, 0:width]
    localheight_map_o = localheight_map_o[0:height, 0:width, :]
    


    num_c, connected_map = cv2.connectedComponents(center_map_o)
    print('num_c:', num_c)
    
    # process component by component
    for cur_cc_idx in range(1, num_c):  # index_0 is the background
        
        if cur_cc_idx % 100 == 0:
            print('processed', str(cur_cc_idx))
            
        centerline_indices = np.where(connected_map == cur_cc_idx)

        centerPoints=[]
        for i, j in zip(centerline_indices[0], centerline_indices[1]):
            if localheight_map_o[i, j, 0] > 0:
                centerPoints.append([i, j])

        if len(centerPoints) == 0:
            continue 

        mini, minj = np.min(centerPoints, axis=0)
        maxi, maxj = np.max(centerPoints, axis=0)

        localheight_result_o = np.zeros((maxi-mini+100, maxj-minj+100, 3), np.uint8)

        for i, j in centerPoints:
            cv2.circle(localheight_result_o, (j-minj+50, i-mini+50), int(localheight_map_o[i][j]*0.4), (0, 0, 255), -1)

        img_gray = cv2.cvtColor(localheight_result_o, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_context = ''

        if len(contours) == 0:
            continue

        for i in range(0, len(contours[0])):
            if i < len(contours[0]) - 1:
                new_context = new_context + str(contours[0][i][0][0].item()+minj-50) + ',' + str(contours[0][i][0][1].item()+mini-50) + ','
            else:
                new_context = new_context + str(contours[0][i][0][0].item()+minj-50) + ',' + str(contours[0][i][0][1].item()+mini-50)

        new_context = new_context + '\n'

        f.writelines(new_context)

    cv2.imwrite(output_path+'prob_' + base_name[0:len(base_name) - 4] + '.jpg', prob_map_o)
    cv2.imwrite(output_path+'cent_' + base_name[0:len(base_name) - 4] + '.jpg', center_map_o)
    cv2.imwrite(output_path+'localheight_map_' + base_name[0:len(base_name) - 4] + '.jpg', localheight_map_o)

    f.close()


    #txt parse
    with open(txt_name, 'r') as f:
        data = f.readlines()

    polyList = []

    for line in data:
        polyStr = line.split(',')
        poly = []
        for i in range(0, len(polyStr)):
            if i % 2 == 0:
                poly.append([int(polyStr[i]), int(polyStr[i+1])])

        polyList.append(poly)


    for i in range(0,len(polyList)):
        polyPoints = np.array([polyList[i]], dtype=np.int32)
        cv2.polylines(map_img, polyPoints, True, (0, 0, 255), 3)


    cv2.imwrite(output_path+'parse_result_'+base_name[0:len(base_name) - 4] + '.jpg',map_img)


    # Generate web annotations: https://www.w3.org/TR/annotation-model/
    annotations = []
    for polygon in polyList:
        svg_polygon_coords = ' '.join([f"{x},{y}" for x, y in polygon])
        annotation = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": "",
            "body": [{
                "type": "TextualBody",
                "purpose": "tagging",
                "value": "null"
            }],
            "target": {
                "selector": [{
                    "type": "SvgSelector",
                    "value": f"<svg><polygon points='{svg_polygon_coords}'></polygon></svg>"
                }]
            }
        }
        annotations.append(annotation)

    with open(output_path+'web_annotations'+base_name[0:len(base_name) - 4] + '.json', 'w') as f:
        f.write(json.dumps(annotations, indent=2))
    # print(f"{polyList}")

print('done processing')


