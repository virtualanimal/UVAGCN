"""
Graph to represent skeleton joints

Joint sequence same as COCO format: {
    0: nose,
    1: left_eye,
    2: right_eye,
    3: left_ear,
    4: right_ear,
    5: left_shoulder,
    6: right_shoulder,
    7: left_elbow,
    8: right_elbow,
    9: left_wrist,
    10: right_wrist,
    11: left_hip,
    12: right_hip,
    13: left_knee,
    14: right_knee,
    15: left_ankle,
    16: right_ankle
}
"""
# 对两人有关系的动作,直接设置自己对自己按照原来的模式，自己对他人设置成0，让模型自己学(或者一个很小的值)


import numpy as np

motion = [9,10,15,16]
body = [9,10,15,16,5,6,11,12,0]

num_node = 34
half_node = num_node//2
self_link = [(i, i) for i in range(num_node)]
inward = [
    (10, 8), (8, 6), (9, 7), (7, 5), # arms
    (15, 13), (13, 11), (16, 14), (14, 12), # legs
    (11, 5), (12, 6), (11, 12), (5, 6), # torso
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears
]
outward = [(j, i) for (i, j) in inward]

inward_two = [(i+half_node,j+half_node) for i,j in inward]
outward_two = [(j, i) for (i, j) in inward_two]

inward_motion = [(one,two+half_node) for one in motion for two in body]  # 第一个人对第二个人
inward_motion_two = [(one+half_node,two) for one in motion for two in body]  # 第二个人对第一个人

inward_motion.extend(inward_motion_two)
outward_motion = [(j, i) for (i, j) in inward_motion]

inward.extend(inward_two)
outward.extend(outward_two)

inward.extend(inward_motion)
outward.extend(outward_motion)

neighbor = inward + outward



def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
