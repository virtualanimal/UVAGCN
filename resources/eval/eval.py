import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='/data/lyp/Skeleton_Based_Action_Recognition/MS-CTR-GCN/work_dir/testB/ctrgcnr_4_modality_avg_ensemble_testB.npy')

if __name__ == "__main__":

    args = parser.parse_args()

    # load label and pred
    label =np.load('/data/lyp/Skeleton_Based_Action_Recognition/resources/data/test_label_A.npy')

    pred = np.load(args.pred_path).argmax(axis=1)
    
    print("pred shape {}".format(pred.shape))
    
    print("lable A shape {}".format(label.max()))

    correct = (pred == label).sum()

    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))