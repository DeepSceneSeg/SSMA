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
import argparse
import cv2

def _int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

def _bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file')
parser.add_argument('-r','--record')

def decode(txt):
    with open(txt) as f:
         all_list=f.readlines()
    file_list=[]
    for line in all_list:
        temp=line.split(' ')
        file_list.append(temp)
    return file_list
     
def convert(f,record_name):
    count = 0
    writer=tf.python_io.TFRecordWriter(record_name)
    for name in f:
        modality1 = cv2.imread(name[0])
        modality2 = cv2.imread(name[1])
        label     = cv2.imread(name[2][:-1],cv2.IMREAD_GRAYSCALE)  
        height    = modality1.shape[0]
        width     = modality1.shape[1]
        modality1 = modality1.tostring() 
        modality2 = modality2.tostring() 
        label     = label.tostring()
        features={'height':_int64_feature(height),
                  'width':_int64_feature(width),
                  'modality1':_bytes_feature(modality1),
                  'label':_bytes_feature(label),
                  'modality2':_bytes_feature(modality2)
                 }
        example=tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        if (count+1)%1==0:
           print 'Processed data: {}'.format(count)
           sys.stdout.flush()    
        count=count+1








def main():
    args = parser.parse_args()
    if args.file:
        file_list=decode(args.file)
    else:
        print '--file file_address missing'
        return
    if args.record:
        record_name=args.record
    else:
        print '--record tfrecord name missing'
        return
    convert(file_list,record_name)

if __name__=='__main__':
        main()
