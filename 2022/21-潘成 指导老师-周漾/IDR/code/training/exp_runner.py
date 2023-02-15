import argparse
import GPUtil
import sys
import jittor as jt
# from idr_train import IDRTrainRunner
# from idr_train import IDRTrainRunner
# from training.idr_train import IDRTrainRunner
from idr_train import IDRTrainRunner
jt.flags.use_cuda = True
if __name__ == '__main__':
    # print('9999999999999')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')

    opt = parser.parse_args()
    print("this is :", __file__, '   line',sys._getframe().f_lineno) 
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
        
    else:
        gpu = opt.gpu

    print(gpu)
    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=gpu,
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 scan_id=opt.scan_id,
                                 train_cameras=opt.train_cameras
                                 )

    trainrunner.run()
    

# python training/exp_runner.py --conf ./confs/dtu_fixed_cameras.conf --scan_id SCAN_ID
