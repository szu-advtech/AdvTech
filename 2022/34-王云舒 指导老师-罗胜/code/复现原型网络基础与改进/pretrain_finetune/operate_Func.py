import numpy as np
import matplotlib.pyplot as plt
def plot_Matrix(epoch, cm, classes, title="confusion matrix of accuracy", cmap=plt.get_cmap('Purples')):
    print("confusion matrix:")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,xlabel='Predicted',ylabel='True_Label')
    ax.set_title("confusion matrix of accuracy")
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor",fontsize=3)
    fmt = 'd'# 标注百分比信息
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('handwrite_result_save/confusion_matrix_epoch_'+str(epoch+1)+'.jpg', dpi=300)
    plt.show()