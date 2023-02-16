import torch.utils.data as tud
import argparse
from Utils import *
from CAVE_Dataset import cave_dataset
import warnings


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/home2/szx/Dataset/DHIF/TREE/Data/Test/', type=str,
                    help='path of the testing data')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=1, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 1)
#归一化

HR_HSI = torch.from_numpy(HR_HSI)
HR_HSI_max = torch.max(HR_HSI)
HR_HSI_min = torch.min(HR_HSI)
HR_HSI = (HR_HSI - HR_HSI_min) / (HR_HSI_max - HR_HSI_min)

HR_MSI = torch.from_numpy(HR_MSI)
HR_MSI_max = torch.max(HR_MSI)
HR_MSI_min = torch.min(HR_MSI)
HR_MSI = (HR_MSI - HR_MSI_min) / (HR_MSI_max - HR_MSI_min)

HR_HSI = HR_HSI.numpy()
HR_MSI = HR_MSI.numpy()

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

model = torch.load("./Checkpoint/f8/Model/model_014.pth")
model = model.eval()
# set the number of parallel GPUs
print("===> Setting GPU")
model = dataparallel(model, 1)



# model = model.cuda()

psnr_total = 0
k = 0
for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(RGB.cuda(), LR.cuda())
        result = out
        result = result.clamp(min=0., max=1.)
    psnr = compare_psnr(result.cpu().detach().numpy(), HR.numpy(), data_range=1.0)
    psnr_total = psnr_total + psnr
    k = k + 1
    print(psnr)
    #
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_path = './Result/ssr/' + str(j + 1) + '.mat'
    sio.savemat(save_path, {'res':res})

print(k)
print("Avg PSNR = %.4f" % (psnr_total / k))
