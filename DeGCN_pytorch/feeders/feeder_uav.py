import random
from matplotlib import use
import numpy as np
import pickle, torch
from . import tools
import random


class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False,tta=0):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self, mmap=False):
        # load label
        # with open(self.label_path, 'rb') as f:
        #     self.sample_name, self.label = pickle.load(f)
        self.label = np.load(self.label_path,)
        # self.sample_name = 'lyp_train'

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

        N, C, T, V, M = self.data.shape
        label_len = len(self.label)
        if label_len != N:
            self.label = [random.randint(0,154) for i in range(N)]
        
        self.num_per_cls_dict = [0,]*(np.array(self.label).max()+1)
        for i in self.label:
            self.num_per_cls_dict[i] += 1
        print(self.num_per_cls_dict)
    
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)   # 每个时间步的每个人的每个样本的均值
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        data_numpy = np.array(data_numpy)

        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # if valid_frame_num == 0:
        #     index = 0
        #     data_numpy = np.array(self.data[index])
        #     data_numpy = np.array(data_numpy)
        #     valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        #
        # # reshape Tx(MVC) to CTVM
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.normalization: # CTVM
            # 把第二个人1号joint移到原点
            # data_numpy[:, :, :, 1] = data_numpy[:, :, :, 1] - data_numpy[:, :, 1:2, 1]

            mean = data_numpy.mean(axis=(2, 3), keepdims=True)
            var = data_numpy.var(axis=(2, 3), keepdims=True)
            data_numpy = (data_numpy - mean) / ((var + 1e-6) ** 0.5)
        if self.random_rot and self.split == 'train':
            data_numpy = tools.random_rot(data_numpy)
            # data_numpy = tools.shear(data_numpy)
            # data_numpy = tools.gaus_noise(data_numpy)

        if self.split == 'test' and self.random_rot:
            if random.random() > 0.5:
                data_numpy = tools.random_rot(data_numpy)


        label = self.label[index]
        if self.bone:
            ntu_pairs = {(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)}

            # ntu_pairs_my = {(3,1),(1,2),(2,4),(4,0),(0,5),
            #                 (5,6),(6,12),(12,11),(11,5),
            #                 (9,7),(7,5),(10,8),(8,6),
            #                 (16,14),(14,12),(15,13),(13,11)}

            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 ] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        # processing
        return data_numpy, label, index
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)