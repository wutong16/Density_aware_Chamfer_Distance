import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random


class ShapeNetH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False, data_set_no=1):
        # if train:
        #     self.input_path = '/mnt/lustre/lijiatong/pl/data/my_train_input_data_denoised_%d.h5' % data_set_no
        #     self.gt_path = '/mnt/lustre/lijiatong/pl/data/my_train_gt_data_%d_fps_%d.h5' % (npoints, data_set_no)
        # else:
        #     self.input_path = '/mnt/lustre/lijiatong/pl/data/my_test_input_data_denoised_%d.h5' % data_set_no
        #     self.gt_path = '/mnt/lustre/lijiatong/pl/data/my_test_gt_data_%d_fps_%d.h5' % (npoints, data_set_no)
        if train:
            self.input_path = '../data/MVP/Train_INPUT.h5'
            self.gt_path = '../data/MVP/Train_GT%d.h5' % (npoints)
        else:
            self.input_path = '../data/MVP/Test_INPUT.h5'
            self.gt_path = '../data/MVP/Test_GT%d.h5' % (npoints)
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel',
                    'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
        self.cat_name = [cat_name[int(i)] for i in np.unique(self.labels)]

        print("tensor shapes: input data {}; gt data {}; labels {}".format(
            self.input_data.shape, self.gt_data.shape, self.labels.shape))
        # print(self.cat_name)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, partial, complete, index

class MVPv1(data.Dataset):
    def __init__(self, train=True, npoints=2048, class_choice=['None'], novel_input=True, novel_input_only=False, data_set_no=1):

        if train:
            self.input_path = '../data/MVP/Train_INPUT.h5'
            self.gt_path = '../data/MVP/Train_GT%d.h5' % (npoints)
            try:
                ucd_file = h5py.File('../data/MVP/Train_INPUT_ucd.h5', 'r')
                self.input_ucd = ucd_file['chair'][()]
            except:
                self.input_ucd = None
        else:
            self.input_path = '../data/MVP/Test_INPUT.h5'
            self.gt_path = '../data/MVP/Test_GT%d.h5' % (npoints)
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel',
                    'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
        self.cat_name = [cat_name[int(i)] for i in np.unique(self.labels)]

        if class_choice[0].lower() == 'none':
            np.random.seed(0)
            indices = np.array(range(self.gt_data.shape[0]))
            print('all class sample size',indices.shape)
            # np.random.shuffle(indices)
            self.index_list = list(indices)

        elif len(class_choice) == 1:
            cat_id = cat_name.index(class_choice[0].lower())
            self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])
            # perm_list = np.random.permutation(len(self.index_list)).tolist()
            # self.index_list = self.index_list[perm_list] # 1D array
            # cat_ls = [self.labels[i] for i in self.index_list]
            # cat_2 = [self.labels[i] for i in perm_list]
        else:
            # multi class, but not all class
            all_cat_indices_selected = []
            for cat in class_choice:
                cat_id = cat_name.index(cat)
                cat_indices = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])
                cat_perm_list = np.random.permutation(len(cat_indices)).tolist()
                cat_indices_selected = cat_indices[cat_perm_list]
                all_cat_indices_selected+=cat_indices_selected.tolist()
                # cat_ls = [self.labels[i] for i in cat_indices_selected]
            # random.shuffle(all_cat_indices_selected)
            self.index_list = np.array(all_cat_indices_selected)


        self.input_data = self.input_data[self.index_list]
        self.gt_data = self.gt_data[self.index_list // 26]
        self.labels = self.labels[self.index_list]
        self.shape_index = self.index_list // 26

        print("tensor shapes: input data {}; gt data {}; labels {}".format(
            self.input_data.shape, self.gt_data.shape, self.labels.shape))
        # print(self.cat_name)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index]))
        label = (self.labels[index])
        shape_index = self.shape_index[index]
        return label, partial, complete, index

class CascadeH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, use_mean_feature=0):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints
        self.train = train
        self.use_mean_feature = use_mean_feature
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if train:
            self.input_path = os.path.join(proj_dir, "data/cascade/train_data.h5")
        else:
            self.input_path = os.path.join(proj_dir, "data/cascade/test_data.h5")
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = input_file['incomplete_pcds'][()]
        self.labels = input_file['labels'][()]
        self.gt_data = input_file['complete_pcds'][()]
        input_file.close()

        if self.use_mean_feature == 1:
            self.mean_feature_path = os.path.join(proj_dir, "data/cascade/mean_feature.h5")
            mean_feature_file = h5py.File(self.mean_feature_path, 'r')
            self.mean_feature = mean_feature_file['mean_features'][()]
            mean_feature_file.close()

        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(np.array(self.input_data[index])).float()
        complete = torch.from_numpy(np.array(self.gt_data[index])).float()
        label = torch.from_numpy(np.array(self.labels[index])).int()
        if self.use_mean_feature == 1:
            mean_feature_input = torch.from_numpy(np.array(self.mean_feature[label])).float()
            return label, partial, complete, mean_feature_input
        else:
            return label, partial, complete

class CascadeShapeNetv1(data.Dataset):

    def __init__(self,split='train',class_choice=['None'],n_samples=0, data_path=None, need_partial=False,min_incomp=0,selected_shapes_idx=None):
        if data_path is None:
            # if data path is not given, take this
            self.DATA_PATH = '../data/cascade'
        else:
            self.DATA_PATH = data_path
        self.split = split
        self.need_partial = need_partial
        # self.catfile = './data/synsetoffset2category.txt'
        if self.split == 'train':
            basename = 'train_data.h5'
        elif self.split == 'test':
            basename = 'test_data.h5'
        elif self.split == 'valid':
            basename = 'valid_data.h5'
        else:
            raise NotImplementedError

        pathname = os.path.join(self.DATA_PATH,basename)
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]

        # import ipdb; ipdb.set_trace()

        np.random.seed(0)
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']
        # print('class choice:',class_choice)
        if class_choice[0].lower() == 'none':
            np.random.seed(0)
            indices = np.array(range(self.gt.shape[0]))
            print('all class sample size',indices.shape)
            np.random.shuffle(indices)
            self.index_list = list(indices)
        elif len(class_choice) == 1:
            cat_id = cat_ordered_list.index(class_choice[0].lower()) 
            self.index = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])
            if n_samples == 0:
                n_samples = self.index.shape[0]
            perm_list = np.random.permutation(len(self.index)).tolist()[:n_samples]
            self.index_list = self.index[perm_list] # 1D array
            cat_ls = [self.labels[i] for i in self.index_list]
            cat_2 = [self.labels[i] for i in perm_list]
        else:
            # multi class, but not all class
            all_cat_indices_selected = []
            for cat in class_choice:
                cat_id = cat_ordered_list.index(cat)
                cat_indices = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])
                cat_perm_list = np.random.permutation(len(cat_indices)).tolist()[:n_samples]
                cat_indices_selected = cat_indices[cat_perm_list]
                all_cat_indices_selected+=cat_indices_selected.tolist()
                cat_ls = [self.labels[i] for i in cat_indices_selected]
            random.shuffle(all_cat_indices_selected)
            self.index_list = np.array(all_cat_indices_selected)

        if min_incomp != 0:
            whitelist = []
            for index in self.index_list:
                incomp = compute_incomp([self.partial[index]],[self.gt[index]])[0]
                if incomp >min_incomp:
                    whitelist.append(index)
            self.index_list = whitelist
        if selected_shapes_idx is not None:
            self.index_list = selected_shapes_idx
        # print('n_samples', len(self.index_list))


    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        if self.need_partial:
            partial = torch.from_numpy(self.partial[full_idx])
        else:
            partial = 0 # placeholder
        return label, partial, gt, full_idx

    def __len__(self):
        return len(self.index_list)

class MultiModalSaved(data.Dataset):

    def __init__(self, split='train', data_path=None, min_incomp=0):
        if data_path is None:
            data_path = '/home/rola/MultiModal-3DShape-Completion/data/mm_output/treegan_baseline_chair_train_real_diverse.h5'
        data = h5py.File(data_path, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.mm_out = data['multi_modal_pcds'][()]
        self.labels = data['labels'][()]
        self.cds = data['cds'][()]
        self.ucds = data['ucds'][()]

        np.random.seed(0)
        indices = np.array(range(self.gt.shape[0]))
        print('all class sample size',indices.shape)
        np.random.shuffle(indices)
        self.index_list = list(indices)

        # if min_incomp != 0:
        #     whitelist = []
        #     for index in self.index_list:
        #         incomp = compute_incomp([self.partial[index]],[self.gt[index]])[0]
        #         if incomp >min_incomp:
        #             whitelist.append(index)
        #     self.index_list = whitelist


    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        mmout = torch.from_numpy(self.mm_out[full_idx]) # fast alr
        label = self.labels[index]
        cd = self.cds[index]
        ucd = self.ucds[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return label, partial, gt, mmout, full_idx, cd, ucd

    def __len__(self):
        return len(self.index_list)