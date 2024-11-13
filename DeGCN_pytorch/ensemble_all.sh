python ensemble_all.py \
--dataset uav-v1 \
--joint-dir "testB_nonorm/ctrgcn_joint/ 20241111 191118/epoch1_test_score.pkl" \
--bone-dir  "testB_nonorm/ctrgcn_bone/ 20241111 191217/epoch1_test_score.pkl" \
--joint-motion-dir "testB/ctrgcn_joint_motion/ 20241108 225544/epoch1_test_score.pkl" \
--bone-motion-dir "testB_nonorm/ctrgcn_bone_motion/ 20241111 191150/epoch1_test_score.pkl" \
--stream5 "testB_nonorm/joint_two/ 20241111 191310/epoch1_test_score.pkl" \
--stream6 ../MS-CTR-GCN/testB/joint/epoch1_test_score.pkl \
--stream7 ../MS-CTR-GCN/testB/bone/epoch1_test_score.pkl \
--stream8 ../MS-CTR-GCN/testB/joint_motion/epoch1_test_score.pkl \
--stream9 ../MS-CTR-GCN/testB/bone_motion/epoch1_test_score.pkl \
--stream10 ../MS-CTR-GCN/testB/joint_two/epoch1_test_score.pkl \
--stream11 ../TE-GCN/testB/joint/epoch0_test_score.pkl \
--stream12 ../TE-GCN/testB/bone/epoch0_test_score.pkl \
--stream13 ../TE-GCN/testB/joint_motion/epoch0_test_score.pkl \
--stream14 ../TE-GCN/testB/bone_motion/epoch0_test_score.pkl \
--stream15 ../TE-GCN/testB/joint_two/epoch0_test_score.pkl \
--stream16  ../SkateFormer/result_B/joint/epoch1_test_score.pkl \
--stream17  ../SkateFormer/result_B/bone/epoch1_test_score.pkl \
--save-ensemble True \
--save_ensemble_ans_file 'merge_best_B.npy'



#--joint-dir "testA_nonorm/ctrgcn_joint/ 20241111 165829/epoch1_test_score.pkl" \
#--bone-dir  "testA_nonorm/ctrgcn_bone/ 20241111 190411/epoch1_test_score.pkl" \
#--joint-motion-dir "testA/ctrgcn_joint_motion/ 20241108 221333/epoch1_test_score.pkl" \
#--bone-motion-dir "testA_nonorm/ctrgcn_bone_motion/ 20241111 165917/epoch1_test_score.pkl" \
#--stream5 "testA_nonorm/joint_two/ 20241111 170010/epoch1_test_score.pkl" \

#--joint-dir "testB_nonorm/ctrgcn_joint/ 20241111 191118/epoch1_test_score.pkl" \
#--bone-dir  "testB_nonorm/ctrgcn_bone/ 20241111 191217/epoch1_test_score.pkl" \
#--joint-motion-dir "testB/ctrgcn_joint_motion/ 20241108 225544/epoch1_test_score.pkl" \
#--bone-motion-dir "testB_nonorm/ctrgcn_bone_motion/ 20241111 191150/epoch1_test_score.pkl" \
#--stream5 "testB_nonorm/joint_two/ 20241111 191310/epoch1_test_score.pkl" \
