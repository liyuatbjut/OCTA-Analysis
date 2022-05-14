from nets.resnet_cls import ResNet50
from PIL import Image
import numpy as np
import random
import copy
import os
import tensorflow as tf
import cv2
from utils import utils
from keras.applications import imagenet_utils
from keras import backend as K
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


class GradCAM:
    def __init__(self):
        self.class_json = ['dr', 'normal']
        self.preprocess_input = imagenet_utils.preprocess_input
        self.target_layer = 'res5c_branch2c'


    def get_class_list(self, preds):
        with open("./dataset/index_word.txt", "r", encoding='utf-8') as f:
            synset = [l for l in f.readlines()]
        max_idx = np.argmax(preds)
        output_idx, output_cls = synset[max_idx].split(';')
        output_cls_list = [output_idx, output_cls, preds[0][max_idx]]

        return output_cls_list


    def processing_image(self, img_path):

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        return x


    def gradcam(self, model, x):

        index = 0
        preds = model.predict(x)
        L = np.argsort(-preds, axis=1)
        pred_class = L[0][index]
        pred_class_list = self.get_class_list(preds)
        pred_class_name = pred_class_list[1]

        pred_output = model.output[:, pred_class]
        last_conv_layer = model.get_layer(self.target_layer)
        grads = K.gradients(pred_output, last_conv_layer.output)[0]
        pooled_grads = K.sum(grads, axis=(0, 1, 2))
        Afeature = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

        pooled_grads_value, conv_layer_output_value = Afeature([x])
        for i in range(pooled_grads_value.shape[0]):
            conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])

        heatmap = np.sum(conv_layer_output_value, axis=-1)

        return heatmap, pred_class_name


    def plot_heatmap(self, heatmap, img_path, pred_class_name, save_result=False, show_result=False):

        img_name_with_suffix = img_path.split('\\')[-1]
        suffix_len = img_name_with_suffix.split('.')[-1].__len__()
        img_name_real = img_name_with_suffix[: -1 * (suffix_len + 1)]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(img_path)
        fig, ax = plt.subplots()
        im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        ax.imshow(im, alpha=0.6)
        ax.imshow(heatmap, cmap='jet', alpha=0.4)

        plt.title(pred_class_name)

        heatmap_saved_root_dir = os.path.join(os.path.abspath('.'), 'heatmap_output')
        if save_result:
            plt.savefig(os.path.join(heatmap_saved_root_dir, img_name_real + '.png'))  # 保存图像
            cv2.imwrite(os.path.join(heatmap_saved_root_dir, 'heat_' + img_name_real + '.png'), heatmap)
        if show_result:
            plt.show()

    def forward(self, model, image, weight, save_result, show_result):
        model.load_weights(weight)
        img = cv2.imread(image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        img = utils.resize_image(img,(224,224))
        pr = model.predict(img)
        pr = np.argmax(pr)

        self.target_layer = 'res5c_branch2c'
        heatmap, pred_class_name = self.gradcam(model, img)
        print(pred_class_name)
        self.plot_heatmap(heatmap, image, pred_class_name, save_result, show_result)


class ClsTester:
    def __init__(self, weight_file, RCLASSES):
        self.RCLASSES = RCLASSES
        self.HEIGHT = 416
        self.WIDTH = 416
        self.log_dir = './logs'
        self.weight_file = weight_file  # weight path
        self.weight_path = os.path.join(self.log_dir, self.weight_file)
        self.test_txt_path = ".\\dataset\\test.txt"
        self.lines = []
        self.cam = GradCAM()
        self.save_points_path = '.\\comparision\\roc_imgs'
        self.points_x_fpr = []
        self.points_y_tpr = []
        self.num_points = 100

    def set_weight(self, weight_file):
        self.weight_file = weight_file  # weight path
        self.weight_path = os.path.join(self.log_dir, self.weight_file)


    def start_testing(self, dataset="test"):
        if dataset not in ("test", "train", "", None):
            print("parameter error of start_testing function")
            exit(-1)

        self.model = ResNet50([224, 224, 3], self.RCLASSES)
        self.model.load_weights(self.weight_path)

        self.test_txt_path = ".\\dataset\\" + dataset + ".txt"

        # prediction
        with open(self.test_txt_path, "r") as f:
            self.lines = f.readlines()

        num_correct = 0
        num_total = self.lines.__len__()

        for name in self.lines:
            img_path = name.split(';')[0]
            y = name.split(';')[1]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            img = utils.resize_image(img, (224, 224))
            pr = self.model.predict(img)
            pr = np.argmax(pr)

            if (utils.print_answer(pr).split(';')[-1].replace('\n', '') == img_path.split('\\')[-2]):
                num_correct += 1

        print("acc: {} with correct: {} / total:{}"
              .format(num_correct / num_total,
                      num_correct,
                      num_total))

    def visualize(self, weight, image, save_result, show_result):
        self.weight_path = os.path.join(self.log_dir, weight)
        self.model = ResNet50([224, 224, 3], self.RCLASSES)
        self.cam.forward(self.model, image, self.weight_path, save_result, show_result)

    def multi_visualize(self, weight, save_result, show_result, dataset="test"):
        if dataset not in ("test", "train", "", None):
            print("parameter error of start_testing function")
            exit(-1)

        self.test_txt_path = ".\\dataset\\" + dataset + ".txt"

        with open(self.test_txt_path, "r") as f:
            self.lines = f.readlines()

        self.weight_path = os.path.join(self.log_dir, weight)
        self.model = ResNet50([224, 224, 3], self.RCLASSES)

        for name in self.lines:
            img_path = name.split(';')[0]
            self.cam.forward(self.model, img_path, self.weight_path, save_result, show_result)
