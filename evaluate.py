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

import argparse
import datetime
import importlib
import os
import numpy as np
import tensorflow as tf
import yaml
from dataset.helper import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_test.config')

def test_func(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    data_list, iterator = get_test_data(config)
    model = model_func(num_classes=config['num_classes'], training=False)
    images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
    images1_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
    model.build_graph(images_pl, images1_pl)
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:', len(import_variables)
    saver = tf.train.Saver(import_variables)
    saver.restore(sess, config['checkpoint'])
    sess.run(iterator.initializer)
    step = 0
    total_num = 0
    output_matrix = np.zeros([config['num_classes'], 3])
    while 1:
    	try:
            img, label, img1 = sess.run([data_list[0], data_list[1], data_list[2]])
            feed_dict = {images_pl : img, images1_pl : img1}
            probabilities = sess.run([model.softmax], feed_dict=feed_dict)
            prediction = np.argmax(probabilities[0], 3)
            gt = np.argmax(label, 3)
            prediction[gt == 0] = 0
            output_matrix = compute_output_matrix(gt, prediction, output_matrix)
            total_num += label.shape[0]
            if (step+1) % config['skip_step'] == 0:
                print '%s %s] %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), total_num)
                print 'mIoU: ', compute_iou(output_matrix)

            step += 1

        except tf.errors.OutOfRangeError:
            print 'mIoU: ', compute_iou(output_matrix), 'total_data: ', total_num
            break

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print '--config config_file_address missing'
    test_func(config)

if __name__ == '__main__':
    main()
