export CUDA_VISIBLE_DEVICES=1
python main.py --config config/uav/test/joint.yaml --work-dir testB_nonorm/ctrgcn_joint --save-score True \
    --phase "test"  --weights  "work_dir/ws_300_nonorm/ctrgcn_joint/20241110 212551/epoch_70_36540.pt"

python main.py --config config/uav/test/bone_motion.yaml --work-dir testB_nonorm/ctrgcn_bone_motion --save-score True \
    --phase "test"  --weights  "work_dir/ws_300_nonorm/ctrgcn_bone_motion/20241111 064830/epoch_72_37584.pt"

python main.py --config config/uav/test/bone.yaml --work-dir testB_nonorm/ctrgcn_bone --save-score True \
    --phase "test"  --weights  "work_dir/ws_300_nonorm/ctrgcn_bone/20241111 092747/epoch_74_38628.pt"

python main.py --config config/uav/test/joint_motion.yaml --work-dir testB_nonorm/ctrgcn_joint_motion --save-score True \
    --phase "test"  --weights  "work_dir/ws_300_nonorm/ctrgcn_joint_motion/20241110 221025/epoch_80_41760.pt"

python main.py --config config/uav/test/joint_two.yaml --work-dir testB_nonorm/joint_two --save-score True \
    --phase "test"  --weights  "work_dir/ws_300_nonorm/ctrgcn_joint_two_ws300/20241111 072253/epoch_67_34974.pt"




