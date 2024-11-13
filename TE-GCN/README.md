# TE-GCN

Code for the paper ["Temporal-Enhanced Graph Convolution Network for Skeleton-based Action Recognition"](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12086)


Please cite the following paper if you use this repository in your reseach.
```
@article{xie_tegcn_2022,
author = {Xie, Yulai and Zhang, Yang and Ren, Fang},
doi = {https://doi.org/10.1049/cvi2.12086},
journal = {IET Comput.Vis.},
title = {{Temporal-enhanced graph convolution network for skeleton-based action recognition}},
year = {2022}
}
```

Note that:
- This code is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- This code is from https://github.com/xieyulai/TE-GCN

## Simple Introduction

​	我们使用了TEGCN的五个模态，包括 joint 、bone、 joint_motion、bone_motion、joint_two[joint模态，只是建图方式不一样]。

​	可以通过[`python gen_model.py`](../resources/gen_modal.py)获得由joint模态变换后的其他模态(我们对赛方给的bone关系做了一定修改)

## TRAIN

You can train the your model using the scripts:
```
sh scripts/train.sh
```

## TEST
You can test the your model using the scripts:
```
sh scripts/test.sh
```

## Ensemble

在得到四个模态的推理结果之后可以通过运行以下指令去得到后融合多模态的结果

```
sh ensemble_tegcn.sh
```

## WEIGHTS

我们将通过四个模态训练的五个模型的的权重文件放在了[google drive](https://drive.google.com/drive/folders/1sQglM8_eU3qquLz6vTDUYCJceoNOHCoR?usp=drive_link)
