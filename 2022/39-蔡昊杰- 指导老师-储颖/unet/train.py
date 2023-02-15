from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from data import *
from utils import *
#from evaluation import *
from torchvision.utils import save_image
#import cv2


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 0是表示从0开始
        image, label = data
        device = torch.device(
            "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
        image, label = image.to(device), label.to(device)  # 数据放进GPU里
        opt.zero_grad()  # 优化器参数清零

        #forword+backward+update
        image = image.type(torch.FloatTensor)  # 转化数据类型,不转则会报错
        image = image.to(device)
        outputs = net(image)
        loss = loss_func(outputs, label.long())  # 进行loss计算

        lll = label.long().cpu().numpy()  # 把label从GPU放进CPU

        loss.backward(retain_graph=True)  # 反向传播(求导)
        opt.step()  # 优化器更新model权重

        running_loss += loss.item()  # 收集loss的值

        if batch_idx % 100 == 99:
            print('[epoch: %d,idex: %2d] loss:%.3f' %
                  (epoch+1, batch_idx+1, running_loss/322))  # 训练集的数量,可根据数据集调整
            runing_loss = 0.0  # 收集的loss值清零

        torch.save(net.state_dict(),
                   f='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth')  # 保存权重


if __name__ == "__main__":
    print(os.getcwd())
    #torch.multiprocessing.set_start_method('spawn')

    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    pv_trn = PVDataset(train_image_path, train_label_path)

    train_loader = DataLoader(
        dataset=pv_trn, batch_size=batch_size, shuffle=True)     #num_workers  = 4

    loss_func = nn.BCEWithLogitsLoss()
    #metric = SegmentationMetric(2)

    #showDataset(train_loader, batch_size)
    net = Unet(3, 2)

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)     # 优化器
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    device = torch.device(
        "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
    print(device)
    net.to(device)  # 网络放入GPU里加速
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')

    loss_list = []
    acc_list = []
    for epoch in range(epochs):  # while True:
        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            label = label[:, :2, :, :, ]   # 截取label的1个通道(2, 2,200,200)
            #label = torch.argmax(label, dim=1).float()   # (2,3,200,200)->(2,200,200)
            #label = torch.unsqueeze(label, 1)           # 升维 ((2, 200,200)->(1, 2,200,200))
            #label = torch.squeeze(label, dim=1)

            out = net(image)       #(2,2,200,200)
            #tmp = torch.argmax(out, dim=1).float()      # (2,200, 200)
            #out.requires_grad = True
            #out = torch.unsqueeze(out, 1)      # (2,200,200)->(2,1,200,200)

            opt.zero_grad()
            #printTensor2Txt(out.cpu(), 'out')
            #printTensor2Txt(label.cpu(), 'label')
            train_loss = loss_func(
                out, label)
            # tensor->numpy
            '''
            tmp1, tmp2 = (out*255).detach().numpy(), (label *255).detach().numpy()
            tmp1 = np.asarray(tmp1.astype(np.uint8), order="C")
            tmp2 = np.asarray(tmp2.astype(np.uint8), order="C")
            hist = metric.addBatch(tmp1, tmp2)
            IoU = metric.IntersectionOverUnion()
            '''

            
            train_loss.backward()
            opt.step()

            if i % 10 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                #print(f'--train_loss===>>{IoU.item()}')
            if i % (len(train_loader)-1) == 0 and i != 0:
                torch.save(net.state_dict(), weight_path)
                loss_list.append(train_loss.item())
                #acc_list.append()
                print('save weight successfully')

            _image = image[0]
            _label = label[0]     #(2,2,200,200)->(2, 200 ,200)
            _out = out[0]  # (2,200,200)->(200,200)
            _label = _label[:1, :, :, ]     #(2,200,200)->(1, 200 ,200),否则不能使用expand
            _out = _out[:1, :, :, ]
            print(_out.shape)
            #_out = torch.unsqueeze(_out, dim=0)

            # 截取label的1个通道(2,200,200)->(1,200,200)
            #_label = torch.squeeze(_label[:1, :, :, ], dim=0)    # 维度为的数量1才可去掉!

            _label =_label.expand(3, 200, 200)      # 将维度拓展 (1, 200, 200)->(3, 200, 200),只有该维度的数量为1才可以expand
            _out = _out.expand(3, 200, 200)      
            img = torch.stack([_image, _label, _out], dim=0)
            save_image(img, f'{save_img_path}/{i}.tif')

    with open(trn_loss_path, 'a') as f1:
        f1.write(str(loss_list) + '\n')     # 换行

        #with open(trn_acc_path, 'w') as f2:
            #f2.write(str(acc_list))
