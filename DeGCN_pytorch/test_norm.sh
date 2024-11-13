export CUDA_VISIBLE_DEVICES=1
python main.py --config config/uav/test/joint.yaml --work-dir testB/ctrgcn_joint --save-score True \
    --phase "test"  --weights  "work_dir/ws_300/ctrgcn_joint/ 20241107 220938/epoch_71_37062.pt"

python main.py --config config/uav/test/bone_motion.yaml --work-dir testB/ctrgcn_bone_motion --save-score True \
    --phase "test"  --weights  "work_dir/ws_300/ctrgcn_bone_motion/ 20241108 073214/epoch_74_38628.pt"

python main.py --config config/uav/test/bone.yaml --work-dir testB/ctrgcn_bone --save-score True \
    --phase "test"  --weights  "work_dir/ws_300/ctrgcn_bone/ 20241108 224317/epoch_72_37584.pt"

python main.py --config config/uav/test/joint_motion.yaml --work-dir testB/ctrgcn_joint_motion --save-score True \
    --phase "test"  --weights  "work_dir/ws_300/ctrgcn_joint_motion/ 20241108 123733/epoch_67_34974.pt"

python main.py --config config/uav/test/joint_two.yaml --work-dir testB/joint_two --save-score True \
    --phase "test"  --weights  "work_dir/uav2/ctrgcn_joint_two_ws300/ 20241107 173414/epoch_77_40194.pt"




