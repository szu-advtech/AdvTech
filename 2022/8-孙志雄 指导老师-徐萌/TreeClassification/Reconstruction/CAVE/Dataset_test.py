"""
@Name: Dataset_test.py
@Auth: SniperIN_IKBear
@Date: 2022/11/22-12:22
@Desc: 
@Ver : 0.0.0
"""
import torch.utils.data as tud
from Utils import *
import torch



class cave_dataset(tud.Dataset):
    def __init__(self, opt, HR_HSI, HR_MSI, istrain=False):
        super(cave_dataset, self).__init__()
        self.path = opt.data_path
        self.istrain = istrain
        self.factor = opt.sf
        if istrain:
            self.num = opt.trainset_num
            self.file_num = 1
            self.sizeIx = opt.sizeIx
            self.sizeIy = opt.sizeIy
        else:
            self.num = opt.testset_num #########
            self.file_num = 1
            self.sizeIx = opt.sizeIx
            self.sizeIy = opt.sizeIy
        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI
    #看不懂这里，这里是在下采样吗？
    def H_z(self, z, factor, fft_B):
        f = torch.rfft(z, 2, onesided=False)  # 原始
        # f = torch.fft.fft2(z, dim=(-2, -1))  # [1, 31, 96, 96]
        # f = torch.stack((f.real, f.imag), -1)
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            Hz = torch.irfft(M, 2, onesided=False)
            x = Hz[:, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            Hz = torch.irfft(M, 2, onesided=False)  # 原始
            # Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1)).real
            x = Hz[:, :, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        return x

    def __getitem__(self, index):
        if self.istrain == True:
            index1 = random.randint(0, self.file_num - 1)
        else:
            index1 = index

        sigma = 2.0
        HR_HSI = self.HR_HSI[:, :, :, index1]
        HR_MSI = self.HR_MSI[:, :, :, index1]

        sz = [self.sizeIx, self.sizeIy]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px = random.randint(0, 2560 - self.sizeIx)
        py = random.randint(0, 2560 - self.sizeIy)
        hr_hsi = HR_HSI[px:px + self.sizeIx:1, py:py + self.sizeIy:1, :]
        hr_msi = HR_MSI[px:px + self.sizeIx:1, py:py + self.sizeIy:1, :]

        if self.istrain == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi = np.rot90(hr_hsi)
                hr_msi = np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2, 0, 1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)

        return lr_hsi, hr_msi, hr_hsi

    def __len__(self):
        return self.num
