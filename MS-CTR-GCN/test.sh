export CUDA_VISIBLE_DEVICES=3

python main.py \
--config config/uav/test/joint_test.yaml \
--weights  work_dir/uav2/ctrgcn_joint/runs-37-9657.pt \
--work-dir testB/joint \
--save-score True \
--phase "test"
#
#
python main.py \
--config config/uav/test/bone_test.yaml \
--weights  work_dir/uav2/ctrgcn_bone/runs-38-9918.pt \
--work-dir testB/bone \
--save-score True \
--phase "test"
#
python main.py \
--config config/uav/test/joint_motion_test.yaml \
--weights  work_dir/uav2/ctrgcn_joint_motion/runs-38-9918.pt \
--work-dir testB/joint_motion \
--save-score True \
--phase "test"

#
python main.py \
--config config/uav/test/bone_motion_test.yaml \
--weights work_dir/uav2/ctrgcn_bone_motion/runs-43-11223.pt \
--work-dir testB/bone_motion \
--save-score True \
--phase "test"


python main.py \
--config config/uav/test_two/joint_test.yaml \
--weights  work_dir/train_two/ctrgcn_joint/runs-41-10701.pt \
--work-dir testB/joint_two \
--save-score True \
--phase "test"

