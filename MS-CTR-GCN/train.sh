export CUDA_VISIBLE_DEVICES=4
python main.py --config config/uav/train/joint.yaml --work-dir work_dir/uav2/ctrgcn_joint
python main.py --config config/uav/train/bone_motion.yaml --work-dir work_dir/uav2/ctrgcn_bone_motion
python main.py --config config/uav/train/bone.yaml --work-dir work_dir/uav2/ctrgcn_bone
python main.py --config config/uav/train/joint_motion.yaml --work-dir work_dir/uav2/ctrgcn_joint_motion

python main.py --config config/uav/train_two/joint.yaml --work-dir work_dir/train_two/ctrgcn_joint




