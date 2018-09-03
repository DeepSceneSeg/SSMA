''' SSMA:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation
 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''



import tensorflow as tf
import numpy as np
def get_train_batch(config):
    filenames = [config['train_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    if config['dataset']=='forest':
    	dataset = dataset.map(lambda x:parser_forest(x,config['num_classes']))
    else:
        dataset = dataset.map(lambda x:parser(x,config['num_classes']))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.repeat(100)
    dataset=dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator
def get_train_data(config):
    iterator = get_train_batch(config)
    if config['type']=='rgb':
            dataA,label,dataB,dataC = iterator.get_next()
    elif config['type']=='jet':
            dataB,label,dataA,dataC = iterator.get_next()
    elif config['type']=='hha':
            dataC,label,dataB,dataA = iterator.get_next()
    elif config['type']=='rgb_jet':
            dataA,label,dataB,dataC = iterator.get_next()
    elif config['type']=='rgb_hha':
            dataA,label,dataC,dataB = iterator.get_next()
    else:
            print 'Not known type :',config['type']
            print 'Available_types : rgb, jet, hha, rgb_jet,rgb_hha'
    return [dataA,label,dataB,dataC],iterator

def get_test_data(config):
    iterator = get_test_batch(config)
    if config['type']=='rgb':
            dataA,label,dataB,dataC = iterator.get_next()
    elif config['type']=='jet':
            dataB,label,dataA,dataC = iterator.get_next()
    elif config['type']=='hha':
            dataC,label,dataB,dataA = iterator.get_next()
    elif config['type']=='rgb_jet':
            dataA,label,dataB,dataC = iterator.get_next()
    elif config['type']=='rgb_hha':
            dataA,label,dataC,dataB = iterator.get_next()
    else:
            print 'Not known type :',config['type']
            print 'Available_types : rgb, jet, hha, rgb_jet,rgb_hha'
    return [dataA,label,dataB,dataC],iterator

def get_test_batch(config):
    filenames = [config['test_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    if config['dataset']=='forest':
    	dataset = dataset.map(lambda x:parser_forest(x,config['num_classes']))
    else:
        dataset = dataset.map(lambda x:parser(x,config['num_classes']))
    dataset = dataset.batch(config['batch_size'])
    #dataset=dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    return iterator

def compute_output_matrix(label_max, pred_max, output_matrix):
    for i in xrange(output_matrix.shape[0]):
        temp=pred_max==i
        temp_l=label_max==i
        tp=np.logical_and(temp,temp_l)
        temp[temp_l]=True
        fp=np.logical_xor(temp,temp_l)
        temp=pred_max==i
        temp[fp]=False
        fn=np.logical_xor(temp,temp_l)
        output_matrix[i,0]+=np.sum(tp)
        output_matrix[i,1]+=np.sum(fp)
        output_matrix[i,2]+=np.sum(fn)
    return output_matrix

def compute_iou(output_matrix):
    return np.sum(output_matrix[1:,0]/(np.sum(output_matrix[1:,:],1).astype(np.float32)+1e-10))/(output_matrix.shape[0]-1)*100
def parser_forest(proto_data,num_classes):
    
    features={'id':tf.FixedLenFeature((), tf.string, default_value=""),
             'channel_depth':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_rgb':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_label':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_evi':tf.FixedLenFeature((), tf.int64, default_value=0),
             'height':tf.FixedLenFeature((), tf.int64, default_value=0),
             'width':tf.FixedLenFeature((), tf.int64, default_value=0),
             'img_raw':tf.FixedLenFeature((), tf.string, default_value=""),
             'label':tf.FixedLenFeature((), tf.string, default_value=""),
             'depth_raw':tf.FixedLenFeature((), tf.string, default_value=""),
             'evi_raw':tf.FixedLenFeature((), tf.string, default_value=""),
             }
    parsed_features = tf.parse_single_example(proto_data, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    label = tf.decode_raw(parsed_features['label'], tf.uint8)
    depth = tf.decode_raw(parsed_features['depth_raw'], tf.uint8)
    evi = tf.decode_raw(parsed_features['evi_raw'], tf.uint8)
    height=tf.cast(parsed_features['height'], tf.int32)
    width=tf.cast(parsed_features['width'], tf.int32)
    channel_rgb=tf.cast(parsed_features['channel_rgb'], tf.int32)
    channel_depth=tf.cast(parsed_features['channel_depth'], tf.int32)
    channel_evi=tf.cast(parsed_features['channel_evi'], tf.int32)
    channel_label=tf.cast(parsed_features['channel_label'], tf.int32)
    
    label=tf.reshape(label,[height,width,channel_label])
    label=tf.one_hot(label,num_classes)
    label=tf.squeeze(label, axis=2)
    
    image=tf.reshape(image,[channel_rgb,height,width])
    image=tf.transpose(image,[1,2,0])
    
    depth=tf.reshape(depth,[channel_depth,height,width])
    depth=tf.transpose(depth,[1,2,0])
    
    evi=tf.reshape(evi,[channel_evi,height,width])
    evi=tf.transpose(evi,[1,2,0])

    return tf.cast(image,tf.float32),tf.cast(label,tf.float32),tf.cast(depth,tf.float32),tf.cast(evi,tf.float32)
def parser(proto_data,num_classes):
    
    features={'id':tf.FixedLenFeature((), tf.string, default_value=""),
             'channel_depth':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_depth_hha':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_rgb':tf.FixedLenFeature((), tf.int64, default_value=0),
             'channel_label':tf.FixedLenFeature((), tf.int64, default_value=0),
             'height':tf.FixedLenFeature((), tf.int64, default_value=0),
             'width':tf.FixedLenFeature((), tf.int64, default_value=0),
             'img_raw':tf.FixedLenFeature((), tf.string, default_value=""),
             'label':tf.FixedLenFeature((), tf.string, default_value=""),
             'depth_raw':tf.FixedLenFeature((), tf.string, default_value=""),
             'depth_hha':tf.FixedLenFeature((), tf.string, default_value="")
             }
    parsed_features = tf.parse_single_example(proto_data, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    label = tf.decode_raw(parsed_features['label'], tf.uint8)
    depth = tf.decode_raw(parsed_features['depth_raw'], tf.uint8)
    hha = tf.decode_raw(parsed_features['depth_hha'], tf.uint8)
    height=tf.cast(parsed_features['height'], tf.int32)
    width=tf.cast(parsed_features['width'], tf.int32)
    channel_rgb=tf.cast(parsed_features['channel_rgb'], tf.int32)
    channel_depth=tf.cast(parsed_features['channel_depth'], tf.int32)
    channel_label=tf.cast(parsed_features['channel_label'], tf.int32)
    height=384
    width=768
    label=tf.reshape(label,[height,width,channel_label])
    label=tf.one_hot(label,num_classes)
    label=tf.squeeze(label, axis=2)
    
    image=tf.reshape(image,[channel_rgb,height,width])
    image=tf.transpose(image,[1,2,0])
    
    depth=tf.reshape(depth,[channel_depth,height,width])
    depth=tf.transpose(depth,[1,2,0])
    
    hha=tf.reshape(hha,[channel_depth,height,width])
    hha=tf.transpose(hha,[1,2,0])
    

    return tf.cast(image,tf.float32),tf.cast(label,tf.int32),tf.cast(depth,tf.float32),tf.cast(hha,tf.float32)
