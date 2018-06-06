from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import sys
from PIL import Image
import os
import shutil
import cv2

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter="", dtype=[('col1', 'S120'), ('col2', 'i8')])
    #f=open(train_file)
    #lines=f.readlines()
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels)

def extract_image(filename,  resize_height, resize_width):
    image = cv2.imread(filename,0)
    #print(image)
    #if image==None:
        #return 'none'
    image = cv2.resize(image, (resize_height, resize_width))
    #b,g,r = cv2.split(image)
    #rgb_image = cv2.merge([r,g,b])
    return image

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(image_paths, labels, out_path, size=(150,150)):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    print("Converting: " + out_path)
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images-1)
            # Load the image-file using matplotlib's imread function.
            img = Image.open(path)
            img = img.resize(size)
            img = np.array(img)
            # Convert the image to raw bytes.
            img_bytes = img.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }
            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)
            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)
            # Serialize the data.
            serialized = example.SerializeToString()
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

if __name__=='__main__':
    base_dir = 'k-fold'

    out_path = base_dir + "/tfrecords"
    for i in range(26):
        train_file = base_dir +"/path_txt"+"/train%02d.txt"%(i+1)
        valid_file = base_dir + "/path_txt"+"/valid%02d.txt"%(i+1)

        path_tfrecords_train = out_path+"/train%02d.tfrecords"%(i+1)
        path_tfrecords_valid = out_path+"/test%02d.tfrecords"%(i+1)

        examples_train,labels_train=load_file(train_file)
        examples_valid,labels_valid=load_file(valid_file)
        convert(examples_train,labels_train,path_tfrecords_train)
        convert(examples_valid,labels_valid,path_tfrecords_valid)