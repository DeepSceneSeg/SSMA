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
import network_base
from ssma_helper import expert

class SSMA(network_base.Network):
    def __init__(self, num_classes=12, learning_rate=0.001, float_type=tf.float32, weight_decay=0.0005,
                 decay_steps=30000, power=0.9, training=True, ignore_label=True, global_step=0,
                 has_aux_loss=False, expert_not_fixed=False):
        super(SSMA, self).__init__()
        self.model1 = expert(training=expert_not_fixed, num_classes=num_classes) 
        self.model2 = expert(training=expert_not_fixed, num_classes=num_classes) 
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initializer = 'he'
        self.has_aux_loss = has_aux_loss
        self.float_type = float_type
        self.power = power
        self.decay_steps = decay_steps
        self.training = training
        self.bn_decay_ = 0.99
        self.eAspp_rate = [3, 6, 12]
        self.residual_units = [3, 4, 6, 3]
        self.filters = [256, 512, 1024, 2048]
        self.strides = [1, 2, 2, 1]
        self.global_step = global_step
        
        if ignore_label:
            self.weights = tf.ones(self.num_classes-1)
            self.weights = tf.concat((tf.zeros(1), self.weights), 0)
        else:
            self.weights = tf.ones(self.num_classes)
        
    def _setup(self):   
        self.in1 = tf.concat([self.model1.eAspp_out, self.model2.eAspp_out], 3)
        self.skip1_in1 = tf.concat([self.model1.skip2, self.model2.skip2], 3)
        self.skip2_in1 = tf.concat([self.model1.skip1, self.model2.skip1], 3)

        self.ssma_red = tf.nn.relu(self.conv_bias(self.in1, 3, 1, 32, 'conv511'))
        self.ssma_expand = tf.nn.sigmoid(self.conv_bias(self.ssma_red, 3, 1, 512, 'conv512')) 
        
        self.ssma_skip1_red = tf.nn.relu(self.conv_bias(self.skip1_in1, 3, 1, 8, 'conv513'))
        self.ssma_skip1_expand = tf.nn.sigmoid(self.conv_bias(self.ssma_skip1_red, 3, 1, 48, 'conv514'))
 
        self.ssma_skip2_red = tf.nn.relu(self.conv_bias(self.skip2_in1, 3, 1, 8, 'conv518'))
        self.ssma_skip2_expand = tf.nn.sigmoid(self.conv_bias(self.ssma_skip2_red, 3, 1, 48, 'conv519'))

        self.in2 = tf.multiply(self.in1, self.ssma_expand)
        self.skip1_in2 = tf.multiply(self.skip1_in1, self.ssma_skip1_expand)
        self.skip2_in2 = tf.multiply(self.skip2_in1, self.ssma_skip2_expand)
        
        self.in3 = self.conv_batchN_relu(self.in2, 1, 1, 256, 'conv515', relu=False)
        self.skip1_out = self.conv_bias(self.skip1_in2, 3, 1, 24, 'conv552')
        self.skip2_out = self.conv_bias(self.skip2_in2, 3, 1, 24, 'conv553'

        ### Upsample/Decoder
        with tf.variable_scope('conv41'):
             self.deconv_up1 = self.tconv2d(self.in3, 4, 256, 2)
             self.deconv_up1 = self.batch_norm(self.deconv_up1)
        x = tf.expand_dims(tf.expand_dims(tf.reduce_mean(self.deconv_up1, [1,2]) ,1) ,2)
        x = self.conv_batchN_relu(x, 1, 1, 24, 'conv550')
        self.side_out1 = tf.multiply(x, self.skip1_out)
        self.up1 = self.conv_batchN_relu(tf.concat((self.deconv_up1, self.side_out1), 3), 3, 1, 256, name='conv89') 
        self.up1 = self.conv_batchN_relu(self.up1, 3, 1, 256, name='conv96')
        with tf.variable_scope('conv16'):
             self.deconv_up2 = self.tconv2d(self.up1, 4, 256, 2)
             self.deconv_up2 = self.batch_norm(self.deconv_up2)
        x = tf.expand_dims(tf.expand_dims(tf.reduce_mean(self.deconv_up2, [1, 2]), 1), 2)
        x = self.conv_batchN_relu(x, 1, 1, 24, 'conv551')
        self.side_out2 = tf.multiply(x, self.skip2_out)
        self.up2 = self.conv_batchN_relu(tf.concat((self.deconv_up2, self.side_out2), 3), 3, 1, 256, name='conv88') 
        self.up2 = self.conv_batchN_relu(self.up2, 3, 1, 256,name='conv95')
        self.up2 = self.conv_batchN_relu(self.up2, 1, 1, self.num_classes, name='conv78')
        with tf.variable_scope('conv5'):
             self.deconv_up3 = self.tconv2d(self.up2, 8, self.num_classes, 4)
             self.deconv_up3 = self.batch_norm(self.deconv_up3)      
            
        self.softmax = tf.nn.softmax(self.deconv_up3)
        ## Auxilary
        if self.has_aux_loss:
           self.aux1 = tf.nn.softmax(tf.image.resize_images(self.conv_batchN_relu(self.deconv_up2, 1, 1, self.num_classes, name='conv911', relu=False),
                                                            [self.input_shape[1], self.input_shape[2]]))
           self.aux2 = tf.nn.softmax(tf.image.resize_images(self.conv_batchN_relu(self.deconv_up1 , 1, 1, self.num_classes, name='conv912', relu=False),
                                                            [self.input_shape[1], self.input_shape[2]]))
               
    def _create_loss(self,label):
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.softmax+1e-10), self.weights), axis=[3]))
        if self.has_aux_loss:
           aux_loss1 = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.aux1+1e-10), self.weights), axis=[3]))
           aux_loss2 = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.aux2+1e-10), self.weights), axis=[3]))
           self.loss = self.loss+0.6*aux_loss1+0.5*aux_loss2
    
    def create_optimizer(self):
        self.lr = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                            self.decay_steps, power=self.power)
        
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def build_graph(self, data, data1, label=None):
        with tf.variable_scope('rgb/resnet_v1_50'):
            self.model1.build_graph(data)
        with tf.variable_scope('depth/resnet_v1_50'):
            self.model2.build_graph(data1)
        self._setup()
        if self.training:
           self._create_loss(label)
          
def main():
    print "Do Nothing"
   
if __name__ == '__main__':
    main()

