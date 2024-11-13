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
    parser.add_argument('--stream9', default=None)
    parser.add_argument('--stream10', default=None)
    parser.add_argument('--stream11', default=None)
    parser.add_argument('--stream12', default=None)
    parser.add_argument('--stream13', default=None)
    parser.add_argument('--stream14', default=None)
    parser.add_argument('--stream15', default=None)
    parser.add_argument('--stream16', default=None)
    parser.add_argument('--stream17', default=None)
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
            r6 = list(pickle.load(r6))
    if arg.stream7 is not None:
        with open(arg.stream7, 'rb') as r7:
            r7 = list(pickle.load(r7))
    if arg.stream8 is not None:
        with open(arg.stream8, 'rb') as r8:
            r8 = list(pickle.load(r8))
    if arg.stream9 is not None:
        with open(arg.stream9, 'rb') as r9:
            r9 = list(pickle.load(r9))
    if arg.stream10 is not None:
        with open(arg.stream6, 'rb') as r10:
            r10 = list(pickle.load(r10))
    if arg.stream11 is not None:
        with open(arg.stream11, 'rb') as r11:
            r11 = list(pickle.load(r11).items())
    if arg.stream12 is not None:
        with open(arg.stream12, 'rb') as r12:
            r12 = list(pickle.load(r12).items())
    if arg.stream13 is not None:
        with open(arg.stream13, 'rb') as r13:
            r13 = list(pickle.load(r13).items())
    if arg.stream14 is not None:
        with open(arg.stream14, 'rb') as r14:
            r14 = list(pickle.load(r14).items())
    if arg.stream15 is not None:
        with open(arg.stream15, 'rb') as r15:
            r15 = list(pickle.load(r15).items())
    if arg.stream16 is not None:
        with open(arg.stream16, 'rb') as r16:
            r16 = list(pickle.load(r16).items())
    if arg.stream17 is not None:
        with open(arg.stream17, 'rb') as r17:
            r17 = list(pickle.load(r17).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0
    # if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:

    weight_options  = np.arange(0, 2.5, 0.5)
    best_alpha = None
    best_acc = 0.0

    # arg.alpha = [1.7, 1, 0.6, 0.6, 1.5]
    # arg.alpha = [1.7, 0,0,0, 1.5]
    # arg.alpha = [1,1.2,0.1,0,1.2] + [1,1.3,0.1,0,0.9] + [1,0.,0,0,0] + [0]
    # arg.alpha = [1.8,2.6,0.2,0,2.4] + [2,2.6,0.2,0.0,1.9] + [1.9,0.,0,0,0.] + [0,0]

    # arg.alpha = [1.8, 2.6, 0.2, 0, 2.4] + [2, 2.6, 0.2, 0, 1.9] + [1.9, 0., 0.1, 0, 0.] + [0.2, 0.1] # last best
    # arg.alpha = [3.6, 5.2, 0.4, 0, 4.8] + [4, 5.6, 0.4, 0, 3.8] + [3.8, 0., 0.2, 0, 0.] + [0.4, 0.2]

    arg.alpha = [2, 1.8, 0.5, 0.2, 1.4] + [2,1.5,0.3,0.2,1] + [1.3, 0.8, 0.2, 0.2, 1] + [1, 1]


    save_ensemble_ans = []
    save_ensemble_ans_file = arg.save_ensemble_ans_file
    for i in tqdm(range(len(r1))):
        if i >= len(label):
            l = label[len(label) - 1]
        else:
            l = label[i]
        _,k11 = r1[i]
        _,k22 = r2[i]
        _,k33 = r3[i]
        _,k44 = r4[i]
        _, k55 = r5[i]

        k66 = r6[i]
        k77 = r7[i]
        k88 = r8[i]
        k99 = r9[i]
        k1010 = r10[i]

        _, k1111 = r11[i]
        _, k1212 = r12[i]
        _, k1313 = r13[i]
        _, k1414 = r14[i]
        _, k1515 = r15[i]

        _, k1616 = r16[i]
        _, k1717 = r17[i]

        r = k11 * arg.alpha[0] + k22 * arg.alpha[1] + k33 * arg.alpha[2] + k44 * arg.alpha[3] + k55 * arg.alpha[4]
        r += k66 * arg.alpha[5] + k77 * arg.alpha[6] + k88 * arg.alpha[7] + k99 * arg.alpha[8] + k1010 * arg.alpha[9]
        r += k1111 * arg.alpha[10] + k1212 * arg.alpha[11] + k1313 * arg.alpha[12] + k1414 * arg.alpha[13] + k1515 * arg.alpha[14]
        r += k1616 * arg.alpha[15] + k1717 * arg.alpha[16]

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

