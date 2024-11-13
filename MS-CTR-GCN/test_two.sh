export CUDA_VISIBLE_DEVICES=6

#python main.py \
#--config config/uav/test_two/joint_test.yaml \
#--weights  work_dir/train_two/ctrgcn_joint/runs-41-10701.pt \
#--work-dir testB/joint_two \
#--save-score True \
#--phase "test"
##
##
#python main.py \
#--config config/uav/test_two/bone_test.yaml \
#--weights  work_dir/train_two/ctrgcn_bone/runs-38-9918.pt \
#--work-dir testB/bone_two \
#--save-score True \
#--phase "test"
#
python main.py \
--config config/uav/test_two/joint_motion_test.yaml \
--weights  work_dir/train_two/ctrgcn_joint_motion/runs-38-9918.pt \
--work-dir testA/joint_motion_two \
--save-score True \
--phase "test"

#
#python main.py \
#--config config/uav/test_two/bone_motion_test.yaml \
#--weights work_dir/train_two/ctrgcn_bone_motion/runs-42-10962.pt \
#--work-dir testB/bone_motion_two \
#--save-score True \
#--phase "test"

#python main.py \
#--config config/uav/joint_bone_test.yaml \
#--weights  /data/lyp/Skeleton_Based_Action_Recognition/MS-CTR-GCN/work_dir/uav/joint_bone_double_sp/runs-38-19494.pt \
#--work-dir work_dir/testB/joint_bone_test \
#--save-score True \
#--phase "test"

