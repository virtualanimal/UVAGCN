export CUDA_VISIBLE_DEVICES=5
#python main.py --config config/uav/train_two/joint.yaml --work-dir work_dir/train_two/ctrgcn_joint
#python main.py --config config/uav/train_two/bone_motion.yaml --work-dir work_dir/train_two/ctrgcn_bone_motion
#python main.py --config config/uav/train_two/bone.yaml --work-dir work_dir/train_two/ctrgcn_bone
python main.py --config config/uav/train_two/joint_motion.yaml --work-dir work_dir/train_two/ctrgcn_joint_motion




