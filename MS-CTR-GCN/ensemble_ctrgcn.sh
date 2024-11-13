python ensemble_ctrgcn.py \
--dataset uav-v1 \
--joint-dir testB/joint/epoch1_test_score.pkl \
--bone-dir testB/bone/epoch1_test_score.pkl \
--joint-motion-dir testB/joint_motion/epoch1_test_score.pkl \
--bone-motion-dir testB/bone_motion/epoch1_test_score.pkl \
--stream5  testB/joint_two/epoch1_test_score.pkl \
--save-ensemble True \
--save_ensemble_ans_file 'ctrgcnr_5_modality_ensemble_testB.npy'



