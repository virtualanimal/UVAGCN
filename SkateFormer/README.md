
# SkateFormer

Code for the paper ["SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition"]


## Requirements
Python >= 3.9.16

PyTorch >= 1.12.1

Platforms: Ubuntu 22.04, CUDA 11.6




## Data Preparation
You can prepare data using the scripts:
```bash
python test_label.py
python uav_25.py
```


## Training
You can train the your model using the scripts:
```bash
sh train.sh
```


## Testing
You can train the your model using the scripts:

```bash
sh test.sh
```


## Weights
我们将权重文件放在了[Google Drive](https://drive.google.com/drive/folders/1meK-PG5fA9-Llh7fBEsR8zHJt4F6V8jh?usp=drive_link)上, 你可以把它下载到你的[work_dir](.\work_dir)下