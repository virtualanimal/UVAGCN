export CUDA_VISIBLE_DEVICES=3
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main.py --config config/compete_data/test/joint.yaml --work-dir testB/joint --seed 777 \
  --weights work_dir/joint/best-49-14900.pt

python main.py --config config/compete_data/test/bone.yaml --work-dir testB/bone --seed 777 \
  --weights work_dir/bone/best-37-11324.pt

python main.py --config config/compete_data/test/joint_motion.yaml --work-dir testB/joint_motion --seed 777 \
  --weights work_dir/joint_motion/best-32-9834.pt

python main.py --config config/compete_data/test/bone_motion.yaml --work-dir testB/bone_motion --seed 777 \
  --weights work_dir/bone_motion/best-37-11324.pt

python main.py --config config/compete_data/two/test/joint.yaml --work-dir testB/joint_two --seed 777 \
  --weights two_w_value/best/joint-30-9238.pt