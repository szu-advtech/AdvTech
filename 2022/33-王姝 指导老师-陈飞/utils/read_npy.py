import numpy as np
# tests=np.load('./results/informer_CSI300_ftMS_sl30_ll15_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.npy')
trues=np.load('./results/informer_CSI300_ftMS_sl30_ll15_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/true.npy')
trues=trues.reshape(-1,7)
np.savetxt('./results/informer_CSI300_ftMS_sl30_ll15_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/true.txt',trues)
preds=np.load('./results/informer_CSI300_ftMS_sl30_ll15_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.npy')
preds=preds.reshape(-1,7)
np.savetxt('./results/informer_CSI300_ftMS_sl30_ll15_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.txt',preds)

