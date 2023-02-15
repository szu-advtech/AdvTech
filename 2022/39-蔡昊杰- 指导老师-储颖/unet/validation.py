from data import *
from train import *
from utils import *

if __name__ == "__main__":
    print(os.getcwd())

    imageList = os.listdir(validation_image_path)
    labelList = os.listdir(validation_label_path)
    pv_val = PVDataset(validation_image_path, validation_label_path)

    val_loader = DataLoader(
        dataset=pv_val, batch_size=1, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    #metric = SegmentationMetric(2)

    #showDataset(train_loader, batch_size)
    net = Unet(3, 3)

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)     # 优化器
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    device = torch.device(
        "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
    net.to(device)  # 网络放入GPU里加速
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')

    epoch = 1
    while epoch < len(val_loader):
        loss_list = []
        acc_list = []
        for i, (image, label) in enumerate(val_loader):
            image, label = image.to(device), label.to(device)

            out = net(image)    # (2,2,200,200)->(2,200,200)

            train_loss = F.binary_cross_entropy(
                Variable(out, requires_grad=True), Variable(label, requires_grad=True))

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 10 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                #print(f'--train_loss===>>{IoU.item()}')
            if i % 40 == 0 and i != 0:
                torch.save(net.state_dict(), weight_path)
                loss_list.append(train_loss.item())
                print('save weight successfully')
            _image = image[0]
            _label = label[0]
            _out = out[0]
            img = torch.stack([_image, _label, _out], dim=0)
            save_image(img, f'{save_val_img_path}/{i}.tif')

        epoch += 1
        with open(val_loss_path, 'a') as f1:  # 写入(文件中若有数据，写入内容，会把原来的覆盖)
            f1.write(str(loss_list) + '\n')     # 换行

        #with open(trn_acc_path, 'w') as f2:
            #f2.write(str(acc_list))
