export CUDA_VISIBLE_DEVICES=7

python main.py --config config/uav/train/norm/joint.yaml --work-dir work_dir/ws_300/ctrgcn_joint
python main.py --config config/uav/train/norm/bone_motion.yaml --work-dir work_dir/ws_300/ctrgcn_bone_motion
python main.py --config config/uav/train/norm/bone.yaml --work-dir work_dir/ws_300/ctrgcn_bone
python main.py --config config/uav/train/norm/joint_motion.yaml --work-dir work_dir/ws_300/ctrgcn_joint_motion
python main.py --config config/uav/train/norm/joint_two.yaml --work-dir work_dir/uav2/ctrgcn_joint_two_ws300



