import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class LossHistory():

    """
    :param result_dir:在这个项目上面跑的所有结果都会保存在该目录下
    :param model：需要进行运行的模型
    :param dataset：因为可能会跑多个数据集，所以传入数据集名称，为每个数据集创建目录
    :param hyper_para：在文件中保存所有的超参数
    :param 输入图像的维度信息，用于绘制模型图
    :return
    """

    def __init__(self, hyp_params, tuing_params):
        self.epoch_train_loss        = []
        self.val_loss     = []
        self.test_loss  = []

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = getattr(models, hyp_params.model+'Model')(hyp_params)
        # self.model = self.model.to(device)
        self.result_dir = hyp_params.log_dir
        self.dataset = hyp_params.dataset

        # 保存所有结果的 results
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # 保存当前数据集所有结果的
        time_str = str(datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'))
        whether_aligned = "_aligned_" if hyp_params.aligned else "_noAligned_"
        self.dataset_result_path = os.path.join(self.result_dir, self.dataset + whether_aligned + time_str)
        if not os.path.exists(self.dataset_result_path):
            os.makedirs(self.dataset_result_path)

        # 保存模型和权重的
        self.model_save_path = os.path.join(self.dataset_result_path, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 保存tensorboard数据的
        self.tensorboard_dir = os.path.join(self.dataset_result_path, 'tensorboard_dir')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)

        # 保存图片和txt文件的
        self.logs_dir = os.path.join(self.dataset_result_path, 'logs')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.loss_txt = os.path.join(self.logs_dir, "epoch_result" + time_str + ".txt")

        self.writer     = SummaryWriter(self.tensorboard_dir)


        self.tuing_params = tuing_params
        self.write_hyper_params()

    def append_loss(self, epoch, epoch_train_loss, val_loss, test_loss):
        # epoch_train_loss, val_loss, test_loss
        self.epoch_train_loss.append(epoch_train_loss)
        self.val_loss.append(val_loss)
        self.test_loss.append(test_loss)

        finished_epoch_time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d  %H:%M:%S')
        with open(self.loss_txt, 'a') as f:
            f.write(finished_epoch_time_str + "  Epoch-" + str(epoch) + ":  epoch_train_loss=" + str(epoch_train_loss))
            f.write("  val_loss=" + str(val_loss))
            f.write("  test_loss=" + str(test_loss))
            f.write("\n")

        self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        self.writer.add_scalar('Loss/valid', val_loss, epoch)
        self.writer.add_scalar('Loss/test', test_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.epoch_train_loss))

        # 绘图前我们要先创建一个Figure对象，Figure对象是一个空白区域。通过pyplot包中的figure函数进行创建。
        plt.figure()
        # 传入x y 通过plot绘制出折线图, linewidth：线条宽度。label：图例标签。
        plt.plot(iters, self.epoch_train_loss, 'red', linewidth = 2, label='epoch_train_loss')
        plt.plot(iters, self.val_loss, 'green', linewidth = 2, label='val loss')
        plt.plot(iters, self.test_loss, '#FFFF00', linewidth = 2, label='test loss')
        # 使用 pyplot 中的 grid() 方法来设置图表中的网格线
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss_original')
        # plt.legend()函数的作用是给图像加图例。
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.logs_dir, "loss_original.png"))
        # plt.cla() # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
        plt.cla()
        plt.close("all")

        # 绘图前我们要先创建一个Figure对象，Figure对象是一个空白区域。通过pyplot包中的figure函数进行创建。
        plt.figure()
        try:
            if len(self.epoch_train_loss) < 25:
                num = 5
            else:
                num = 15
            # scipy.signal.savgol_filter(x, window_length, polyorder)尽量将折现平滑化，而且折现的形状保持不变。
            # window_length 对window_length内的数据进行多项式拟合。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
            # polyorder为多项式拟合的阶数.它越小，则平滑效果越明显；越大，则更贴近原始曲线。
            plt.plot(iters, scipy.signal.savgol_filter(self.epoch_train_loss, num, 3), '#B22222', linestyle='--',
                     linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#228B22', linestyle='--', linewidth=2,
                     label='smooth val loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.test_loss, num, 3), '#EEDD82', linestyle='--', linewidth=2,
                     label='smooth test loss')

        except:
            pass
        # 使用 pyplot 中的 grid() 方法来设置图表中的网格线
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss_smooth')
        # plt.legend()函数的作用是给图像加图例。
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.logs_dir, "loss_smooth.png"))
        # plt.cla() # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
        plt.cla()
        plt.close("all")


    def write_hyper_params(self):
        with open(self.loss_txt, 'a') as f:
            f.write('Configurations:\n')
            f.write('-' * 70 + '\n')
            f.write('%25s  %40s' % ('keys', 'values') + '\n')
            f.write('-' * 70 + '\n')
            for key, value in self.tuing_params.items():
                f.write('%25s  %40s' % (str(key), str(value)) + '\n')
            f.write('-' * 70)
            f.write("\n\n")