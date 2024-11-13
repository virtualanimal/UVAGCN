export CUDA_VISIBLE_DEVICES=7

python main.py --config config/uav/train/joint.yaml --work-dir work_dir/ws_300_nonorm/ctrgcn_joint
python main.py --config config/uav/train/bone_motion.yaml --work-dir work_dir/ws_300_nonorm/ctrgcn_bone_motion
python main.py --config config/uav/train/bone.yaml --work-dir work_dir/ws_300_nonorm/ctrgcn_bone
python main.py --config config/uav/train/joint_motion.yaml --work-dir work_dir/ws_300_nonorm/ctrgcn_joint_motion
python main.py --config config/uav/train/two_train/joint.yaml --work-dir work_dir/ws_300_nonorm/ctrgcn_joint_two_ws300



