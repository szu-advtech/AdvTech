"""
@Name: PlotPred.py
@Auth: SniperIN_IKBear
@Date: 2022/12/2-16:13
@Desc: 
@Ver : 0.0.0
"""
import torch
import matplotlib.pyplot as plt


def plot(pred, acc,h):
    img = pred.indices.reshape(100, 100)
    img = torch.squeeze(img).detach().cpu().numpy()
    plt.imshow(img)
    plt.title(f'Target Classfication{h}m')
    plt.text(x=16,  # 文本x轴坐标
             y=48,  # 文本y轴坐标
             s=f'准确率为：{acc}',  # 文本内容

             color='r',

             # 添加文字背景色
             bbox={'facecolor': '#74C476',  # 填充色
                   'edgecolor': 'b',  # 外框色
                   'alpha': 0.5,  # 框透明度
                   'pad': 8,  # 本文与框周围距离
                   }

             )
    plt.savefig(r'./result.jpg')
    plt.show()


# plt.title('0.8_iron_classfication')