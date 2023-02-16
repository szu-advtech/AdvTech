import os
from tqdm import tqdm
import numpy as np
import torch
import data
from option import args
from models import *
import utility
#python test.py --ckp_path ../experiment/sa/edsr/x4/model/model_latest.pt --TS S --scale 4 --model EDSR --n_resgroups 10 --n_resblocks 6
# python train.py --scale 4 --teacher [EDSR] --model EDSR --alpha 0.5 --feature_loss_used 1 --feature_distilation_type 10*test --features [1,2,3] --epochs 100 --save_results --chop --patch_size 192 --ckp_dir ours/edsrwockernel/x4
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]

def test():
    model.eval()
    with torch.no_grad():     
        for idx_data, d in enumerate(test_loader):
            for idx_scale, scale in enumerate(args.scale):
                d.dataset.set_scale(idx_scale)
                psnr_list = []
                psnr = 0
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = prepare(lr, hr)
                    fms, sr = model(lr)
                    sr = utility.quantize(sr, args.rgb_range)
                    
                    save_list = [sr]
                    psnr += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)


                psnr /= len(d)
                psnr_list.append(psnr)
                best_psnr = max(psnr_list)
                print('[{} x{}]\tPSNR: {:.3f}'.format(
                        d.dataset.name,
                        scale,
                        best_psnr
                    ))

   
    
if __name__ == '__main__':
    loader = data.Data(args)
    test_loader = loader.loader_test
       
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    device = torch.device('cpu' if args.cpu else 'cuda')
    
    if args.model == 'EDSR':
        model = edsr.EDSR(args).to(device)
    elif args.model == 'RCAN':
        model = rcan.RCAN(args).to(device)
    elif args.model =='SAN':
        model = san.SAN(args).to(device)
    
    if args.TS == 'T':
        model.load_state_dict_teacher(torch.load(args.ckp_path))
    elif args.TS == 'S':
        model.load_state_dict_student(torch.load(args.ckp_path))

    test()