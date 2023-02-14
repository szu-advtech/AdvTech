# test.py
import torch,argparse
import net,config
from torchvision.datasets import CIFAR10
import cv2


def show_CIFAR10(index):
    eval_dataset=CIFAR10(root='dataset', train=False, download=False)
    print(eval_dataset.__len__())
    print(eval_dataset.class_to_idx,eval_dataset.classes)
    img, target=eval_dataset[index][0], eval_dataset[index][1]

    import matplotlib.pyplot as plt
    plt.figure(str(target))
    plt.imshow(img)
    plt.show()


def test(args):
    classes={'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    index2class=[x  for x in classes.keys()]
    print("calss:",index2class)

    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    transform = config.test_transform

    ori_img=cv2.imread(args.img_path,1)
    img=cv2.resize(ori_img,(32,32)) # evry important，influence the result

    img=transform(img).unsqueeze(dim=0).to(DEVICE)

    model=net.SimCLRStage2(num_class=10).to(DEVICE)
    model.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=False)

    pred = model(img)

    prediction = torch.argsort(pred, dim=-1, descending=True)

    label=index2class[prediction[:, 0:1].item()]
    cv2.putText(ori_img,"this is "+label,(30,30),cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0), 1)
    cv2.imshow(label,ori_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # show_CIFAR10(2)

    parser = argparse.ArgumentParser(description='test SimCLR')
    parser.add_argument('--pre_model', default=config.pre_model, type=str, help='')
    parser.add_argument('--img_path', default=".//imgs//bird2.jpg", type=str, help='')

    args = parser.parse_args()
    test(args)
