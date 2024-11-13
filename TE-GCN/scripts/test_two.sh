export CUDA_VISIBLE_DEVICES=3

#python main.py --config config/compete_data/two/test/joint.yaml --work-dir testB/joint_two --seed 777 \
#  --weights /data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/two_w_value/joint/joint-30-9238.pt

python main.py --config config/compete_data/two/test/bone.yaml --work-dir testA/bone_two --seed 777 \
  --weights /data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/two_w_value/bone/best-31-9536.pt

python main.py --config config/compete_data/two/test/joint_motion.yaml --work-dir testA/joint_motion_two --seed 777 \
  --weights /data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/two_w_value/joint_motion/best-34-10430.pt

python main.py --config config/compete_data/two/test/bone_motion.yaml --work-dir testA/bone_motion_two --seed 777 \
  --weights /data/lyp/Skeleton_Based_Action_Recognition/TE-GCN/two_w_value/bone_motion/best-59-17880.pt