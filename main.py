import os
from train.cls_trainer import ClsTrainer
from train.cls_tester import ClsTester
from dataset.generate_txt import GenerateTxt

class WholeAlgorithmTrainer:
    def __init__(self):
        self.TRAINS = 'trains'

        # two sorts of images
        self.IMAGE_TYPE_3M = 1
        self.IMAGE_TYPE_6M = 0

        # settings
        self.data_type = self.TRAINS  # dataset type, only one here for the example
        self.current_epoch = 1
        self.predict_weight = 'trains_batch_1_ep045-loss0.107-val_loss0.372.h5'
        self.visual_weight = self.predict_weight
        self.vis_img = os.path.join(os.path.abspath('.'), 'dataset\\trains\\DR\\10005.bmp')
        self.classes = 2  # number of classes
        self.distinguish_data = False  # 3M和6M images analyzed respectively
        self.image_type = self.IMAGE_TYPE_6M
        self.save_visual_result = False
        self.show_visual_result = True
        self.epochs = 50

        # init
        self.root_path = os.path.abspath('.')
        self.data_path = os.path.join(self.root_path, 'dataset')
        self.trainer = ClsTrainer(RCLASSES=self.classes, EPOCHS=self.epochs)
        self.tester = ClsTester(weight_file='', RCLASSES=self.classes)
        self.txt_generator = GenerateTxt(current_epoch=self.current_epoch,
                                         root_path=self.data_path,
                                         folder_name=self.data_type,
                                         distinguish=self.distinguish_data,
                                         image_type=self.image_type)


    def forward(self):
        # set txt file of dataset
        batch = 1  # this batch is not the batch_size. It's for the cross validation, from 1 to 6
        self.txt_generator.set_epoch(batch)
        self.txt_generator.forward()


        # train
        self.trainer.start_training(batch=batch, data_type=self.data_type)


        # # test / predict
        # self.tester.set_weight(self.predict_weight)
        # self.tester.start_testing(dataset="test")


        # # single image visualization
        # self.tester.visualize(weight=self.visual_weight,
        #                       image=self.vis_img,
        #                       save_result=self.save_visual_result,
        #                       show_result=self.show_visual_result)


        # # images visualization for a folder
        # self.tester.multi_visualize(weight=self.visual_weight,
        #                             save_result=True,
        #                             show_result=self.show_visual_result,
        #                             dataset="train")


if __name__ == '__main__':
    wat = WholeAlgorithmTrainer()
    wat.forward()
