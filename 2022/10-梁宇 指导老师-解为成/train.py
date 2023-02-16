from model.miccai_msnet_borad import MSNet, LossNet
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
from utils import dataset_medical
import pandas as pd
from torch.cuda import amp


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:0')

df = pd.DataFrame(columns=['step', 'epoch', 'lr', 'loss1u', 'loss2u', 'loss'])
df.to_csv("", index=False)


def train(Dataset, Network, Network1):

    ## dataset
    train_path = ''
    cfg = Dataset.Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    ## network
    net = Network()
    net1 = Network1()
    net.train(True)
    net1.eval()
    net.to(device)
    net1.to(device)
    torch.backends.cudnn.enabled = False
    for param in net1.parameters():
        param.requires_grad = False
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image = image.to(device).float()
            mask = mask.to(device).float()
            with amp.autocast(enabled=use_fp16):
                output = net(image)
                loss2u = net1(F.sigmoid(output), mask)
                loss1u = structure_loss(output, mask)
                loss = loss1u + 0.1 * loss2u
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if step % cfg.batch == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss1u=%.6f | loss2u=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss1u.item(), loss2u.item(), loss.item()))

                list = [step, epoch+1, optimizer.param_groups[0]['lr'], loss1u.item(), loss2u.item(), loss.item()]
                data = pd.DataFrame([list])
                data.to_csv("log/train_ori.csv", mode='a', header=False, index=False) #mode设为a,就可以向csv文件追加数据了

        if epoch == 50:
            torch.save(net.state_dict(), cfg.savepath+'/model-ori'+str(epoch+1))

        if epoch > cfg.epoch/3*2:
            torch.save(net.state_dict(), cfg.savepath+'/model-ori'+str(epoch+1))


if __name__=='__main__':
    train(dataset_medical, MSNet, LossNet)
