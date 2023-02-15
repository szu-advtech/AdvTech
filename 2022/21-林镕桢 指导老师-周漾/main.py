import sys
from PyQt6.QtWidgets import QApplication
from utils.main_widget import MainWidget
from utils.img_handler import ImageHandler
import torch
from models import Model
import clip


def main():
    model = Model(False)
    model.load("./trained_models/70")
    model.cuda()

    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    img_handler = ImageHandler(model, clip_model, clip_preprocess)

    app = QApplication(sys.argv)
    ex = MainWidget(img_handler)
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()