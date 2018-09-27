''' SSMA:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation
 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import numpy as np
import tensorflow as tf

def get_train_batch(config):
    filenames = [config['train_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parser(x, config['num_classes']))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.repeat(100)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def get_train_data(config):
    iterator = get_train_batch(config)
    dataA, label, dataB = iterator.get_next()
    return [dataA, label, dataB], iterator

def get_test_data(config):
    iterator = get_test_batch(config)
    dataA, label, dataB = iterator.get_next()
    return [dataA, label, dataB], iterator

def get_test_batch(config):
    filenames = [config['test_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parser(x, config['num_classes']))
    dataset = dataset.batch(config['batch_size'])
    iterator = dataset.make_initializable_iterator()
    return iterator

def compute_output_matrix(label_max, pred_max, output_matrix):
    for i in xrange(output_matrix.shape[0]):
        temp = pred_max == i
        temp_l = label_max == i
        tp = np.logical_and(temp, temp_l)
        temp[temp_l] = True
        fp = np.logical_xor(temp, temp_l)
        temp = pred_max == i
        temp[fp] = False
        fn = np.logical_xor(temp, temp_l)
        output_matrix[i, 0] += np.sum(tp)
        output_matrix[i, 1] += np.sum(fp)
        output_matrix[i, 2] += np.sum(fn)

    return output_matrix

def compute_iou(output_matrix):
    return np.sum(output_matrix[1:, 0]/(np.sum(output_matrix[1:, :], 1).astype(np.float32)+1e-10))/(output_matrix.shape[0]-1)*100

def parser(proto_data, num_classes):
    features = {'height':tf.FixedLenFeature((), tf.int64, default_value=0),
                'width':tf.FixedLenFeature((), tf.int64, default_value=0),
                'modality1':tf.FixedLenFeature((), tf.string, default_value=""),
                'label':tf.FixedLenFeature((), tf.string, default_value=""),
                'modality2':tf.FixedLenFeature((), tf.string, default_value="")
               }
    parsed_features = tf.parse_single_example(proto_data, features)
    modality1 = tf.decode_raw(parsed_features['modality1'], tf.uint8)
    label = tf.decode_raw(parsed_features['label'], tf.uint8)
    modality2 = tf.decode_raw(parsed_features['modality2'], tf.uint8)
    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    label = tf.reshape(label, [height, width, 1])
    label = tf.one_hot(label, num_classes)
    label = tf.squeeze(label, axis=2)
    modality1 = tf.reshape(modality1, [height, width, 3])
    modality2 = tf.reshape(modality2, [height, width, 3])
    return tf.cast(modality1, tf.float32), tf.cast(label, tf.int32), tf.cast(modality2, tf.float32)
