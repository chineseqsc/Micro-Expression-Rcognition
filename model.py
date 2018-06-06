from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.python import keras

import os
import shutil

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

def create_model():
    #conv_base = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(150, 150, 3))
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    #print(conv_base.summary())

    # 在conv_base的基础上添加全连接分类网络
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    conv_base.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  metrics=['acc'])

    model_dir = os.path.join(os.getcwd(), "models/vggface_classifier")
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir: ", model_dir)
    vggface_classifier = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                               model_dir=model_dir)
    print(model.summary())

    return vggface_classifier

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

def imgs_input_fn(filenames, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image.set_shape([None, None, None])
        image = tf.image.resize_images(image, [150, 150])
        image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        image.set_shape([150, 150, 3])
        #image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
        d = dict(zip(["vgg16_input"], [image])), label
        return d
    if labels is None:
        labels = [0]*len(filenames)
    labels=np.array(labels)
    # Expand the shape of "labels" if necessory
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=1)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def train():
    acclist = []
    for i in range(26):
        print("*******round%02d**********" % (i + 1))
        base_dir = 'k-fold'
        train_files = base_dir + "/path_txt/train%02d.txt" % (i + 1)
        valid_files = base_dir + "/path_txt/valid%02d.txt" % (i + 1)
        # path_tfrecords_train="k-fold/tfrecords/train%02d.tfrecords"%(i+1)
        # path_tfrecords_valid="k-fold/tfrecords/test%02d.tfrecords"%(i+1)
        examples_train, labels_train = load_file(train_files)
        examples_valid, labels_valid = load_file(valid_files)
        create_model().train(input_fn=lambda: imgs_input_fn(examples_train,
                                                              labels=labels_train,
                                                              perform_shuffle=True,
                                                              repeat_count=1,
                                                              batch_size=20))

        evaluate_results = create_model().evaluate(
            input_fn=lambda: imgs_input_fn(examples_valid,
                                           labels=labels_valid,
                                           perform_shuffle=False,
                                           batch_size=1))
        print("Evaluation results")
        for key in evaluate_results:
            print("   {}, was: {}".format(key, evaluate_results[key]))
            acclist.append(evaluate_results[key])

    print(acclist)

def pretest():
    base_dir = 'k-fold'
    train_files = base_dir + "/path_txt/train01.txt"
    valid_files = base_dir + "/path_txt/valid01.txt"
    #path_tfrecords_train = "k-fold/tfrecords/train11.tfrecords"
    #path_tfrecords_valid = "k-fold/tfrecords/test11.tfrecords"
    examples_train01, labels_train01 = load_file(train_files)
    examples_valid01, labels_valid01 = load_file(valid_files)
    create_model().train(input_fn=lambda: imgs_input_fn(examples_train01,
                                                          labels=labels_train01,
                                                          perform_shuffle=True,
                                                          repeat_count=1,
                                                          batch_size=20))
    evaluate_results = create_model().evaluate(
        input_fn=lambda: imgs_input_fn(examples_valid01,
                                       labels=labels_valid01,
                                       perform_shuffle=False,
                                       batch_size=1))
    print("Evaluation results")
    for key in evaluate_results:
        print("   {}, was: {}".format(key, evaluate_results[key]))

    predict_results = create_model().predict(
        input_fn=lambda: imgs_input_fn(examples_valid01[:10],
                                       labels=None,
                                       perform_shuffle=False,
                                       batch_size=10))

    for i in (predict_results):
        print(i)

    predict_logits = []
    for prediction in predict_results:
        predict_logits.append(prediction['dense_2'][0])
        print(prediction)

    for j in predict_logits:
        print(j)

if __name__=='__main__':
    #create_model()
    pretest()

