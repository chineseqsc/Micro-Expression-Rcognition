#coding：utf-8

import  cv2
import os

import numpy as np
import tensorflow as tf

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

# 对图片进行预处理，将图片转化成神经网络的输入层数据
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

def face_extract(img):
    img = cv.LoadImage("friend1.jpg");

    image_size = cv.GetSize(img)  # 获取图片的大小
    greyscale = cv.CreateImage(image_size, 8, 1)  # 建立一个相同大小的灰度图像
    cv.CvtColor(img, greyscale, cv.CV_BGR2GRAY)  # 将获取的彩色图像，转换成灰度图像
    storage = cv.CreateMemStorage(0)  # 创建一个内存空间，人脸检测是要利用，具体作用不清楚

    cv.EqualizeHist(greyscale, greyscale)  # 将灰度图像直方图均衡化，貌似可以使灰度图像信息量减少，加快检测速度
    # detect objects
    cascade = cv.Load('haarcascade_frontalface_alt2.xml')  # 加载Intel公司的训练库

    # 检测图片中的人脸，并返回一个包含了人脸信息的对象faces
    faces = cv.HaarDetectObjects(greyscale, cascade, storage, 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING,
                                 (50, 50))

    # 获得人脸所在位置的数据
    j = 0  # 记录个数
    for (x, y, w, h), n in faces:
        j += 1
        cv.SetImageROI(img, (x, y, w, h))  # 获取头像的区域
        cv.SaveImage("face" + str(j) + ".jpg", img);  # 保存下来

def face_dect(file_path):
    """
        Detecting faces in image
        :param image:
        :return:  the coordinate of max face
    """
    image = cv2.imread(file_path);
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classifier=cv2.CascadeClassifier("/home/qsc/DEMO/VGGFace_TF/haarcascade_files/haarcascade_frontalface_default.xml")
    faces = classifier.detectMultiScale(image)
    if not len(faces) > 0:
        return None
    max_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face
    face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
    try:
        image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+} Problem during resize")
        return None
    cv2.imshow('img', face_image)

    return face_image


def resize_image(image, size):
    try:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("+} Problem during resize")
        return None
    return image


if __name__=='__main__':
    face_dect('1.jpg')
