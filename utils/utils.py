import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops
import os

def load_image(path):
    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open("./dataset/index_word.txt","r",encoding='utf-8') as f:
        # synset = [l.split(";")[1][:-1] for l in f.readlines()]
        synset = [l for l in f.readlines()]
        
    # print(synset[argmax])
    return synset[argmax]

def get_heatmap_mask():
    root = os.path.dirname(os.path.abspath('.'))
    hm_dir = os.path.join(root, 'get_heatmap_mask')
    hm_path = os.path.join(hm_dir, 'heatmap_mask.png')

    return hm_path


class Evaluator:
    def __init__(self):
        pass

    def forward(self):
        pass

    def other_possibility(self):
        pass