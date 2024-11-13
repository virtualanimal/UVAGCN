import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 记录每个类别的正确和错误数量
correct_class_count = {}
incorrect_class_count = {}

correct_frame_count = {}
incorrect_frame_count = {}

# pre_label_path =  '/data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/test_train/epoch1_test_score.pkl'
# label_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/train_label.npy'
# train_data_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/train_joint.npy'

pre_label_path =  '/data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/work_dir/joint/epoch51_test_score.pkl'
label_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/test_label_A.npy'
train_data_path = '/data/lyp/Skeleton_Based_Action_Recognition/resources/data/test_joint_A.npy'

labels = np.load(label_path)
datas = np.load(train_data_path)
with open(pre_label_path,'rb') as f:
    preds = list(pickle.load(f).items())


N, C, T, V, M = datas.shape
CR =0

for i in tqdm(range(N)):
    label = labels[i]
    _, pred = preds[i]
    pred = np.argmax(pred)
    data = datas[i]
    frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)

    if label == pred:
        CR += 1
        if frame_num in correct_frame_count:
            correct_frame_count[frame_num] += 1
        else:
            correct_frame_count[frame_num] = 1
        if label in correct_class_count:
            correct_class_count[label] += 1
        else:
            correct_class_count[label] = 1
    else:
        if frame_num in incorrect_frame_count:
            incorrect_frame_count[frame_num] += 1
        else:
            incorrect_frame_count[frame_num] = 1
        if label in incorrect_class_count:
            incorrect_class_count[label] += 1
        else:
            incorrect_class_count[label] = 1

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
print(correct_frame_count)
print('*'*20)
print(incorrect_frame_count)

# 绘制预测正确的类别数量分布
axs[0, 0].bar(correct_class_count.keys(), correct_class_count.values(), color='green')
axs[0, 0].set_title('correct class distribution')
axs[0, 0].set_xlabel('class')
axs[0, 0].set_ylabel('count')
axs[0, 0].set_xticks(list(correct_class_count.keys()))
axs[0, 0].set_xticklabels(list(correct_class_count.keys()), rotation=45)

# 绘制预测错误的类别数量分布
axs[0, 1].bar(incorrect_class_count.keys(), incorrect_class_count.values(), color='red')
axs[0, 1].set_title('incorrect class distribution')
axs[0, 1].set_xlabel('class')
axs[0, 1].set_ylabel('count')
axs[0, 1].set_xticks(list(incorrect_class_count.keys()))
axs[0, 1].set_xticklabels(list(incorrect_class_count.keys()), rotation=45)

# 绘制预测正确的数据帧数分布
axs[1, 0].bar(correct_frame_count.keys(), correct_frame_count.values(), color='green')
axs[1, 0].set_title('correct frame distribution')
axs[1, 0].set_xlabel('frame')
axs[1, 0].set_ylabel('count')

# 绘制预测错误的数据帧数分布
axs[1, 1].bar(incorrect_frame_count.keys(), incorrect_frame_count.values(), color='red')
axs[1, 1].set_title('incorrect frame distribution')
axs[1, 1].set_xlabel('frame')
axs[1, 1].set_ylabel('count')

# 调整布局
plt.tight_layout()
plt.savefig('valid_predict_analysis_100.png')

print("Top1:{:.5f}".format(CR/N))

