python ensemble_degcn.py \
--dataset uav-v1 \
--joint-dir "testA_nonorm/ctrgcn_joint/ 20241111 165829/epoch1_test_score.pkl" \
--bone-dir  "testA_nonorm/ctrgcn_bone/ 20241111 190411/epoch1_test_score.pkl" \
--joint-motion-dir "testA/ctrgcn_joint_motion/ 20241108 221333/epoch1_test_score.pkl" \
--bone-motion-dir "testA_nonorm/ctrgcn_bone_motion/ 20241111 165917/epoch1_test_score.pkl" \
--stream5 "testA_nonorm/joint_two/ 20241111 170010/epoch1_test_score.pkl" \
--save-ensemble True \
--save_ensemble_ans_file 'degcn_5_modality_ensemble_testA.npy'



#--joint-dir "testA_nonorm/ctrgcn_joint/ 20241111 165829/epoch1_test_score.pkl" \
#--bone-dir  "testA_nonorm/ctrgcn_bone/ 20241111 190411/epoch1_test_score.pkl" \
#--joint-motion-dir "testA/ctrgcn_joint_motion/ 20241108 221333/epoch1_test_score.pkl" \
#--bone-motion-dir "testA_nonorm/ctrgcn_bone_motion/ 20241111 165917/epoch1_test_score.pkl" \
#--stream5 "testA_nonorm/joint_two/ 20241111 170010/epoch1_test_score.pkl" \


#--joint-dir "testB/ctrgcn_joint/ 20241108 224444/epoch1_test_score.pkl" \
#--bone-dir  "testB/ctrgcn_bone/ 20241110 115415/epoch1_test_score.pkl" \
#--joint-motion-dir "testB/ctrgcn_joint_motion/ 20241108 225544/epoch1_test_score.pkl" \
#--bone-motion-dir "testB/ctrgcn_bone_motion/ 20241108 225446/epoch1_test_score.pkl" \
#--stream5 "testB/joint_two/ 20241108 225618/epoch1_test_score.pkl" \