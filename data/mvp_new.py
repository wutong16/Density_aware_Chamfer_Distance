import torch
import numpy as np
import torch.utils.data as data
import h5py
import os


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = '../data/MVP/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = '../data/MVP/MVP_Test_CP.h5'
        # the hidden test set below is only used for workshop competition
        # elif prefix=="test":
        #     self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]
        cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel',
                    'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
        self.cat_name = [cat_name[int(i)] for i in np.unique(self.labels)]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial
