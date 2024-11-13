#!/bin/bash


EPOCH_NUM=60
BATCH_SIZE=56
WARM_UP=5
SEED=777

export CUDA_VISIBLE_DEVICES=3
python3 main.py --config config/compete_data/train/joint.yaml --work-dir work_dir/joint -model_saved_name work_dir/joint/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/train/bone.yaml --work-dir work_dir/bone -model_saved_name work_dir/bone/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/train/joint_motion.yaml --work-dir work_dir/joint_motion -model_saved_name work_dir/joint_motion/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
python3 main.py --config config/compete_data/train/bone_motion.yaml --work-dir work_dir/bone_motion -model_saved_name work_dir/bone_motion/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED

python3 main.py --config config/compete_data/two/train/joint.yaml --work-dir two_w_value/joint -model_saved_name two_w_value/joint/best --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --num-epoch $EPOCH_NUM --seed $SEED
