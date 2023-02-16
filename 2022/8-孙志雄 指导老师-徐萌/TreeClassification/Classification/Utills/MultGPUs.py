"""
@Name: MultGPUs.py
@Auth: SniperIN_IKBear
@Date: 2022/12/1-10:41
@Desc: 
@Ver : 0.0.0
"""
import torch



def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:

            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model