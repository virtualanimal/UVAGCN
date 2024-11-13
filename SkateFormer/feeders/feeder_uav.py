import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):
        
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()
        if partition:
            self.right_arm = np.array([5, 7, 9]) - 1  # 右肩、右肘、右腕
            self.left_arm = np.array([6, 8, 10]) - 1  # 左肩、左肘、左腕
            self.left_leg = np.array([11, 13, 15]) - 1  # 左髋、左膝、左踝
            self.right_leg = np.array([12, 14, 16]) - 1  # 右髋、右膝、右踝
            self.head = np.array([0, 1, 2, 3, 4]) - 1  # 头部相关关节点
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.head), axis=-1)
            # except for joint no.21

    def load_data(self, mmap=False):
        # data: N C T V M 
        self.label = np.load(self.label_path)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        N, C, T, V, M = self.data.shape
        print(self.data.shape)
        label_len = len(self.label)
        if label_len != N:
            self.label = [random.randint(0, 154) for i in range(N)]  # 若标签数量与数据不一致，随机生成标签

        self.num_per_cls_dict = [0] * (np.array(self.label).max() + 1)
        for i in self.label:
            self.num_per_cls_dict[i] += 1

    def __len__(self):
        return len(self.label)
    
    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        if self.uniform:
            data_numpy, index_t = tools_uav.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                           self.window_size, self.thres)
        else:
            data_numpy, index_t = tools_uav.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)
        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)
            if p < self.intra_p:
                '''内实例增强'''
                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp
                if '1' in self.aug_method:
                    data_numpy = tools_uav.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools_uav.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools_uav.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools_uav.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools_uav.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools_uav.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools_uav.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools_uav.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools_uav.drop_joint(data_numpy, p=0.5)

            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int32)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools_uav.skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()

        # modality
        if self.data_type == 'b':
            j2b = tools_uav.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools_uav.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools_uav.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools_uav.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]
        print(data_numpy.shape)

        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        """计算top-k准确率"""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)