import os

import jittor as jt
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import jittor.transform as transform
from jittor.dataset import CIFAR10
from model.my_PVT_classification import classification as Model

class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}


@jt.no_grad()
def evaluate(model, path):
    model.eval()
    weight_path = './weights/bak/3复杂数据增强model-99.pkl'
    assert os.path.exists(weight_path), "weights file: '{}' not exist.".format(weight_path)
    weights_dict = jt.load(weight_path)
    model.load_state_dict(weights_dict)
    img_list = os.listdir(path=path)
    i = 1
    for img_name in img_list:
        img = os.path.join(path, img_name)
        img = cv.imread(img)
        new_img = img.copy()
        img = jt.array(img).unsqueeze(0).float32()
        pred = model(img)
        pred_class = np.argmax(pred.data, axis=1)[0]

        new_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
        plt.imshow(new_img)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_dict[pred_class], fontsize=36)
        plt.savefig(os.path.join(path, 'out' + str(i) + '.png'))
        i += 1
        plt.show()
        print("类别为：" + class_dict[pred_class])
        input()


from model.config import CONFIGS

config = CONFIGS['PVT_classification_tiny']
if __name__ == '__main__':
    model = Model(config=config)
    path = './data/test'
    evaluate(model, path)
