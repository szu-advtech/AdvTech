import math

import config
import sen12ms_cr_dataLoader
from model.DSen2_CR import DSen2_CR
from data_process.dataIO import get_cloudy_true_predict_img

import torch

from model.value_metric import *


def predict():
    args = config.get_args()

    net = DSen2_CR()
    net = net.eval()

    # 网络初始化
    parameters = torch.load(args.load_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(parameters)

    # 数据集
    dataset = sen12ms_cr_dataLoader.SEN12MSCRDataset(args.predict_data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("数据集初始化完毕，数据集大小为：{}".format(len(dataset)))

    # 将数据装入gpu（在有gpu且使用GPU进行训练的情况下）
    cloud_img = torch.FloatTensor(args.batch_size, 13, 256, 256)
    ground_truth = torch.FloatTensor(args.batch_size, 13, 256, 256)
    csm_img = torch.FloatTensor(args.batch_size, 1, 256, 256)

    """
    if config.use_gpu:
        net = net.cuda()
        cloud_img = cloud_img.cuda()
        ground_truth = ground_truth.cuda()
        csm_img = csm_img.cuda()"""


    with torch.no_grad():
        for iteration, batch in enumerate(dataloader, start=1):
            img_cld, img_csm, img_truth = batch
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)

            img_predict = net(img_cld)

            """print(f"MAE: {cloud_mean_absolute_error(img_truth, img_predict)}", end='\t')
            # print(f"RMSE: {cloud_root_mean_squared_error(img_truth, img_predict)}", end='\t')
            print(f"PSNR: {cloud_psnr(img_truth, img_predict)}", end='\t')
            print(f"SSIM: {ssim((torch.squeeze(img_truth)).numpy(), (torch.squeeze(img_predict)).numpy())}", end='\t')
            print(f"SAM: {cloud_mean_sam(img_truth, img_predict) * 180/math.pi}")"""

            MAE = "{:.3f}".format(cloud_mean_absolute_error(img_truth, img_predict))
            PSNR = "{:.3f}".format(cloud_psnr(img_truth, img_predict))
            SSIM = "{:.3f}".format(get_ssim(img_truth, img_predict))
            SAM = "{:.3f}".format(cloud_mean_sam(img_truth, img_predict) * 180/math.pi)

            str1 = f"MAE: {MAE}\tPSNR: {PSNR}\tSSIM: {SSIM}\tSAM: {SAM}\n"

            f = open('./result.txt', 'a')
            f.write(str1)
            print(iteration)

            print(str1)

            # get_cloudy_true_predict_img(img_cld, img_truth, img_csm, img_predict)

            # SaveImg(output, os.path.join(args.predict_result, "iteration_{}.jpg".format(iteration)))

if __name__ == "__main__":
    predict()
