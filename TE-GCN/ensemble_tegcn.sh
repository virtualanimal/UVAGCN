python ensemble_tegcn.py \
--dataset uav-v1 \
--joint-dir testB/joint/epoch0_test_score.pkl \
--bone-dir testB/bone/epoch0_test_score.pkl \
--joint-motion-dir testB/joint_motion/epoch0_test_score.pkl \
--bone-motion-dir testB/bone_motion/epoch0_test_score.pkl \
--stream5  testB/joint_two/epoch0_test_score.pkl \
--save-ensemble True \
--save_ensemble_ans_file 'tegcn_5_modality_ensemble_testB.npy'