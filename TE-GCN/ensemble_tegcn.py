import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'uav-v1', 'uav-v2'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--stream5', default=None)
    parser.add_argument('--stream6', default=None)
    parser.add_argument('--stream7', default=None)
    parser.add_argument('--stream8', default=None)
    parser.add_argument('--save-ensemble', default=False)
    parser.add_argument('--save_ensemble_ans_file',type=str,default='pred.npy')

    arg = parser.parse_args()

    dataset = arg.dataset
    label = np.load('../resources/data/val_label.npy')

    with open(arg.joint_dir, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.bone_dir, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(arg.joint_motion_dir, 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(arg.bone_motion_dir, 'rb') as r4:
            r4 = list(pickle.load(r4).items())
    if arg.stream5 is not None:
        with open(arg.stream5, 'rb') as r5:
            r5 = list(pickle.load(r5).items())
    if arg.stream6 is not None:
        with open(arg.stream6, 'rb') as r6:
            r6 = list(pickle.load(r6).items())
    if arg.stream7 is not None:
        with open(arg.stream7, 'rb') as r7:
            r7 = list(pickle.load(r7).items())
    if arg.stream6 is not None:
        with open(arg.stream8, 'rb') as r8:
            r8 = list(pickle.load(r8).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0
    # if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:


    # arg.alpha = [1, 1, 0.5 ,0.5]  # ctrgcn
    arg.alpha = [1, 0.8, 0.3, 0.5, 1]
    # arg.alpha = [1, 1, 0.5, 0.5, 0.8]  #tegcn

    save_ensemble_ans = []
    save_ensemble_ans_file = arg.save_ensemble_ans_file
    for i in tqdm(range(len(r1))):
        if i >= len(label):
            l = label[len(label) - 1]
        else:
            l = label[i]
        _,r11 = r1[i]
        _,r22 = r2[i]
        _,r33 = r3[i]
        _,r44 = r4[i]
        _, r55 = r5[i]


        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4]

        save_ensemble_ans.append(r)
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc, arg.alpha)
    if acc>best:
        best = acc
        best_alpha = arg.alpha
    acc5 = right_num_5 / total_num
    print(best, best_alpha)

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    save_ensemble_ans = np.array(save_ensemble_ans)

    if arg.save_ensemble:
        np.save(save_ensemble_ans_file,save_ensemble_ans)
        print('save_ensemble_ans_file')

