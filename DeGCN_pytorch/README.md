# Degcn
​		我们主要是基于[DeGCN_pytorch](https://github.com/WoominM/DeGCN_pytorch) 的代码进行修改的

## Train the model

```bash
sh train.sh
sh train_norm.sh
```
我们会得到多个个模态的参数文件

## Test the model

```bash
sh test.sh
sh test_norm.sh
```
## Multi-stream ensemble
我们训练的最好的模型放在了[google drive](https://drive.google.com/drive/folders/1epS9jb3TzengyGeIN98ytvaOOtiOIamF?usp=drive_link)，你可以下载过来之后放在[./work_dir](./work_dir) 下

在得到四个模态的推理结果之后可以通过运行以下指令去得到后融合多模态的结果

```bash
sh ensemble_degcn.sh
```
## Acknowledgements

Thanks to the original authors for their work!
