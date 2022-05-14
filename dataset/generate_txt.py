import os

class GenerateTxt:
    def __init__(self, current_epoch, root_path, folder_name, distinguish, image_type):
        self.folder_name = folder_name
        self.abspath = root_path
        self.train_txt_path = os.path.join(self.abspath, 'train.txt')
        self.test_txt_path = os.path.join(self.abspath, 'test.txt')
        self.all_txt_path = os.path.join(self.abspath, 'all.txt')
        self.train_imgs_path = os.path.join(self.abspath, self.folder_name)
        self.all_epochs = 6
        self.currect_epoch = current_epoch
        self.distinguish = distinguish
        self.image_type = image_type  # 0 denotes 6M，1 denotes 3M
        self.distinguish_sep = 10300  # 10000~10300 are 6M images，10301~10500 are 3M images


    def generate_the_all_txt(self):
        with open(self.all_txt_path, 'w') as f:
            classes = os.listdir(self.train_imgs_path)
            for index, cls in enumerate(classes):
                new_imgs_list = []
                imgs_list = os.listdir(os.path.join(self.train_imgs_path, cls))
                for im in imgs_list:
                    if self.distinguish:  # True
                        if self.image_type:
                            if int(im.split('.')[0]) > self.distinguish_sep:
                                new_imgs_list.append(os.path.join(self.train_imgs_path, cls, im))
                                f.write(os.path.join(self.train_imgs_path, cls, im) + ";" + str(index) + "\n")
                        else:
                            if int(im.split('.')[0]) <= self.distinguish_sep:
                                new_imgs_list.append(os.path.join(self.train_imgs_path, cls, im))
                                f.write(os.path.join(self.train_imgs_path, cls, im) + ";" + str(index) + "\n")
                    else:
                        new_imgs_list.append(os.path.join(self.train_imgs_path, cls, im))
                        f.write(os.path.join(self.train_imgs_path, cls, im) + ";" + str(index) + "\n")


    def num_imgs_per_cls(self):
        with open(self.all_txt_path, 'r') as f:
            imgs_list = f.readlines()
            # total number of every class
            num_per_class_list = []

            num_classes = int(imgs_list[-1].split(';')[-1].replace('\n', '')) + 1  # number of classes in the dataset
            for cls in range(num_classes):
                num_per_class_val = 0
                for num, img in enumerate(imgs_list):
                    cls_idx = img.split(';')[-1].replace('\n', '')
                    if (str(cls_idx) == str(cls)):
                        num_per_class_val += 1

                num_per_class_list.append(num_per_class_val)

            # number of testing examples of every class
            num_test_img_list = [val // self.all_epochs for val in num_per_class_list]

            num_train_img_list = []  # number of training examples of every class
            for idx in range(num_classes):
                num_train_img_list.append(num_per_class_list[idx] - num_test_img_list[idx])

        return num_per_class_list, num_train_img_list, num_test_img_list

    def generate_txts(self):
        num_per_class_list, num_train_img_list, num_test_img_list = self.num_imgs_per_cls()
        num_classes = num_per_class_list.__len__()
        if self.currect_epoch >= 1:
            current_ep = self.currect_epoch - 1
        else:
            print("current_epoch must be an integer above zero.")
            exit(-1)

        with open(self.all_txt_path, 'r') as fal:
            with open(self.test_txt_path, 'w') as fte:
                with open(self.train_txt_path, 'w') as ftr:

                    # get the list of images of every class
                    imgs_list = fal.readlines()
                    imgs_each_cls_list = []

                    last_res = 0
                    for cls in range(num_classes):
                        imgs_each_cls_list.append([])
                        imgs_each_cls_list[cls].append(imgs_list[last_res: last_res + num_per_class_list[cls]])
                        last_res += num_per_class_list[cls]

                    train_list = []
                    test_list = []

                    # test
                    last_res = [current_ep * val for val in num_test_img_list]

                    for cls in range(num_classes):
                        test_list.append([])
                        test_list[cls].append(imgs_each_cls_list[cls][0][last_res[cls]: last_res[cls] + num_test_img_list[cls]])
                        last_res[cls] += num_test_img_list[cls]

                    avail_test_list = []
                    for cls in range(num_classes):
                        avail_test_list.append(test_list[cls][0])

                    output_test_list = []
                    output_train_list = []
                    test_1dim_list = []

                    for cls in range(num_classes):
                        for ele in avail_test_list[cls]:
                            test_1dim_list.append(ele)

                    for im in imgs_list:
                        if im in test_1dim_list:
                            output_test_list.append(im)
                        else:
                            output_train_list.append(im)

                    for im in output_test_list:
                        fte.write(im)
                    for im in output_train_list:
                        ftr.write(im)
                ftr.close()
            fte.close()
        fal.close()


    def set_epoch(self, epoch):
        self.currect_epoch = epoch

    def forward(self):
        self.generate_the_all_txt()
        self.generate_txts()


if __name__ == '__main__':
    generator = GenerateTxt(6, os.path.abspath('.'))
    generator.forward()
