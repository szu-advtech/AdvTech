import json
import os
import datetime
from sklearn.model_selection import KFold
import torch
from torch.utils import data
import numpy as np

from weight_loss import KpLoss
from eval import evaluate
import transforms
from a_mid_hrnet import HighResolutionNet
from my_dataloaders import Fingerprint


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device.type))

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            # 标准化处理
            #transforms.Normalize(mean=0.4234, std=0.1878)
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=0.4234, std=0.1878)
        ])
    }

    data_root = "./TrainAndTest"
    batch_size = 6

    # 加载数据集的workers number
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)



       
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kth = 0

    for train_index, val_index in kf.split(range(30)):
        kth += 1
        if kth == 1:
            continue
        result_file = "result_fold{}.txt".format(kth)

        print("5-fold-train:", kth)
        print("val: ", val_index)
        
        train_dataset = Fingerprint( transforms=data_transform["train"],dataset="train", train_list=train_index)
        train_data_loader = data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=nw)
        
        val_dataset = Fingerprint( transforms=data_transform["val"], dataset="val", train_list=val_index)
        val_data_loader = data.DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,)
    
        model = HighResolutionNet(num_joints=1)
        model.to(device)
    
        # define optimizer
        lr = 0.001
        weight_decay = 1e-4
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params,
                                      lr=lr,
                                      weight_decay=weight_decay)
    
        # 设置epoch
        epochs = 30
        start_epoch = 1
        
    
        # 从第x个epoch开始训练
        #loadpath = "./path/6.pth"
        #checkpoint = torch.load(loadpath['model'], map_location='cpu')
        #model.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1
        #print("the training process from epoch{}...".format(start_epoch))
    
    
        
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running_loss = 0.0
            # # 损失计算
            mse = KpLoss()
    
            for step, [images, targets] in enumerate(train_data_loader):
                optimizer.zero_grad()
                outputs = model(images.to(device))
                targets.to(device)
                loss =  mse(outputs, targets.to(device))
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
    
            print('[epoch %d] train_loss: %.6f ' %
                  (epoch, running_loss))
            save_path = "./path/"+ f"{kth}" +"/"+ f"{epoch}" + ".pth"
            
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}            
            torch.save(save_files, save_path)
            
            result_file = "result_fold{}.txt".format(kth)
            info = evaluate(model, val_data_loader, device)
            
            with open(result_file, "a") as f:
                save_info = [f"{i:.4f}" for i in info]
                txt = "epoch:{} {}".format(epoch, '  '.join(save_info))
                f.write(txt + "\n")


if __name__ == '__main__':
    main()
