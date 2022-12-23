import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 同态滤波器
def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst

def put(path):
    # image = mpimg.imread(path)
    # image = cv2.imread(os.path.join(base, path), 1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # 针对彩色图片，将RGB图像转换为HSV形式，然后分别对HSV三个通道进行同态滤波，最后组合并转回RGB形式
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # 同态滤波器
    h = homomorphic_filter(H)
    s = homomorphic_filter(S)
    v = homomorphic_filter(V)
    h_image = cv2.merge((h,s,v))
    final = cv2.cvtColor(h_image,cv2.COLOR_HSV2RGB)

    # plt.subplot(121)
    # plt.axis('off')
    # plt.title('原始图像')
    # plt.imshow(image)

    # plt.subplot(122)
    # plt.axis('off')
    # plt.title('同态滤波器图像')
    # plt.imshow(final)

    # # plt.savefig('5.new.jpg')
    # plt.show()
    return final

# 图像处理函数，要传入路径
if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        im = put(img_file)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        cv2.imwrite(new+'/'+filename,im)
