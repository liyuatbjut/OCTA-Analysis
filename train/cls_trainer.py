from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.utils import np_utils
from keras.optimizers import Adam
from nets.resnet_cls import ResNet50
import numpy as np
from utils import utils
import cv2
from keras import backend as K
import tensorflow as tf

K.set_image_dim_ordering('tf')
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


class ClsTrainer:
    def __init__(self, RCLASSES, EPOCHS):
        # 模型保存的位置
        self.log_dir = ".\\logs\\"
        self.train_txt_path = ".\\dataset\\train.txt"
        self.batch_size = 16
        self.lines = []
        self.RCLASSES = RCLASSES  # 分类的总类数
        self.epochs = EPOCHS

    def generate_arrays_from_file(self, lines, batch_size):
        # 获取总长度
        n = len(lines)
        i = 0
        while 1:
            X_train = []
            Y_train = []
            # 获取一个batch_size大小的数据
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(lines)
                name = lines[i].split(';')[0]
                # 从文件中读取图像
                img = cv2.imread(name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255
                X_train.append(img)
                Y_train.append(lines[i].split(';')[1])
                # 读完一个周期后重新开始
                i = (i + 1) % n
            # 处理图像
            X_train = utils.resize_image(X_train, (224, 224))
            X_train = X_train.reshape(-1, 224, 224, 3)

            Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=self.RCLASSES)
            yield (X_train, Y_train)  # 这个是yield

    def start_training(self, batch=0, data_type='None'):
        with open(self.train_txt_path, "r") as f:
            self.lines = f.readlines()
        self.generate_arrays_from_file(self.lines, 4)

        # 打乱行，这个txt主要用于帮助读取数据来训练
        # 打乱的数据更有利于训练
        np.random.seed(12345)
        np.random.shuffle(self.lines)
        np.random.seed(None)

        # 90%用于训练，10%用于估计。
        num_val = int(len(self.lines) * 0.1)

        num_train = len(self.lines) - num_val

        # 建立ResNet50模型
        model = ResNet50([224, 224, 3], self.RCLASSES)
        # model.load_weights(".\\nets\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        #                    by_name=True, skip_mismatch=True)

        # # 指定训练层
        # for i in range(0,len(model.layers)-5):
        #     model.layers[i].trainable = False

        # 保存的方式，3世代保存一次
        checkpoint_period1 = ModelCheckpoint(
            self.log_dir + data_type + '_batch_' + str(batch) + '_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='acc',
            save_weights_only=False,
            save_best_only=False,
            period=1
        )
        # 学习率下降的方式，acc三次不下降就下降学习率继续训练
        reduce_lr = ReduceLROnPlateau(
            monitor='acc',
            factor=0.8,
            patience=10,
            verbose=1
        )
        # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1
        )

        # 交叉熵
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-4),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, self.batch_size))

        # 开始训练
        model.fit_generator(self.generate_arrays_from_file(self.lines[:num_train], self.batch_size),
                            steps_per_epoch=max(1, num_train // self.batch_size),
                            validation_data=self.generate_arrays_from_file(self.lines[num_train:], self.batch_size),
                            validation_steps=max(1, num_val // self.batch_size),
                            epochs=self.epochs,
                            initial_epoch=0,
                            callbacks=[checkpoint_period1, reduce_lr])

        # model.save_weights(self.log_dir + 'last1.h5')

if __name__ == '__main__':
    trainer = ClsTrainer()
    trainer.start_training()
