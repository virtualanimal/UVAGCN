# Multi-Stream CTR-GCN
​		我们主要是基于[CarefreeSun/MS-CTR-GCN](https://github.com/CarefreeSun/MS-CTR-GCN) 的代码进行修改的

## Train the model

We provide scripts for you to train the four basic streams conveniently.
```bash
sh train.sh
```
## Test the model

```bash
sh test.sh
```
## Multi-stream ensemble
我们训练的最好的模型放在了[google drive](https://drive.google.com/drive/folders/1bS42zYV-ClWdX6miC6EaG649p1-e7yRJ?usp=drive_link)，你可以下载过来之后放在[./work_dir](./work_dir) 下

在得到五个模态的推理结果之后可以通过运行以下指令去得到后融合多模态的结果

```bash
sh ensemble_ctrgcn.sh
```
## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

Thanks to the original authors for their work!
