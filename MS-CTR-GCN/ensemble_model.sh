python ensemble_model.py \
--dataset uav-v1 \
--joint-dir ctrgcnr_5_modality_ensemble_testB.npy \
--bone-dir ../TE-GCN/tegcn_5_modality_ensemble_testB.npy \
--joint-motion-dir ../DeGCN_pytorch/degcn_5_modality_ensemble_testB.npy \
--save-ensemble True \
--save_ensemble_ans_file 'model_ensemble_ctrgcn_tegcn_degcn_5_testB.npy'

#--joint-motion-dir  ../UAV-SAR/sttformer/sttformer_4_modality_ensemble_testB_adam.npy \
#--bone-motion-dir  ctrgcnr_4_modality_ensemble_testB.npy \
#--stream5 ../TE-GCN/tegcn_4_modality_ensemble_testB.npy \
#--stream6  ../UAV-SAR/sttformer/sttformer_4_modality_ensemble_testB.npy \