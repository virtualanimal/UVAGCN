#!/bin/bash


EPOCH_NUM=60
BATCH_SIZE=56
WARM_UP=5
SEED=777

export CUDA_VISIBLE_DEVICES=1
#python3 main.py --config config/compete_data/two/train/joint.yaml --work-dir two_w_value/joint -model_saved_name two_w_value/joint/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/two/train/bone.yaml --work-dir two_w_value/bone -model_saved_name two_w_value/bone/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/two/train/joint_motion.yaml --work-dir two_w_value/joint_motion -model_saved_name two_w_value/joint_motion/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/two/train/bone_motion.yaml --work-dir two_w_value/bone_motion -model_saved_name two_w_value/bone_motion/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
