import os
import sys
import time

import cv2
from PyQt5.Qt import *
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QTabWidget, QLabel, QTextEdit, QPushButton, QFileDialog, QMessageBox

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)

image = ""

class AlbumInterface(QWidget):
    def __init__(self,Width = 1600,Height = 800):
        super(AlbumInterface, self).__init__()
        self.Width = Width
        self.Height = Height
        # 主窗口size,之后计算子窗口大小
        self.resize(self.Width, self.Height)
        # 计算左侧窗口size
        self.L_Width, self.L_Height = self.Width - 300, Height
        self.L_imgsize = (self.L_Width - 5, self.L_Height)
        # 计算右侧窗口size
        self.R_Width, self.R_Height = 300, Height
        self.R_imgsize = (self.R_Width - 5, self.R_Height)

        self.scene = ["brandenburg_gate", "lincoln_memorial", "palacio_de_bellas_artes", "pantheon_exterior"]
        AlbumList = list()

        # 创建左边
        self.Left_Widget = QLabel(self)
        self.Left_Widget.resize(self.L_Width, self.L_Height)
        self.Left_Widget.move(0, 0)

        # 创建右边
        self.Right_Widget = QWidget(self)
        self.Right_Widget.resize(self.R_Width, self.R_Height)
        p1 = self.Right_Widget.palette()
        p1.setColor(QPalette.Background,QColor(0,100,100))
        self.Right_Widget.setPalette(p1)
        self.Right_Widget.setAutoFillBackground(True)
        self.Right_Widget.move(self.Left_Widget.pos().x() + self.Left_Widget.width(), 0)

        # 右边子部件
        FormLayout = QFormLayout(self.Right_Widget)
        FormLayout.setVerticalSpacing(20)
        FormLayout.setHorizontalSpacing(15)
        FormLayout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        FormLayout.setLabelAlignment(Qt.AlignBottom | Qt.AlignHCenter)

        FormLayout.addRow(QLabel()) # 添加空行，调整布局
        Label0 = QLabel(self.Right_Widget)
        Label0.setText("数据集场景")
        Label0.setStyleSheet('font-size:40px;')
        Label0.setAlignment(Qt.AlignVCenter)
        Label0.setAlignment(Qt.AlignHCenter)
        Label0.resize(300,100)
        FormLayout.addRow(Label0)
        FormLayout.addRow(QLabel())  # 添加空行，调整布局

        Label1 = QLabel(self.Right_Widget)
        Label1.setText("选择场景")
        self.ComboBox = QComboBox(self)
        self.ComboBox.addItems(self.scene)
        self.ComboBox.view().window().setFixedHeight(150)
        # self.ComboBox.setView(QListView())  # 设置此项后item样式才起作用
        self.ComboBox.setMaxVisibleItems(5)
        # add QCombobox index change event
        self.ComboBox.currentTextChanged.connect(self.ComboBoxChanged)
        FormLayout.addRow(Label1, self.ComboBox)

        button0 = QPushButton(self.Right_Widget)
        button0.resize(200, 60)
        button0.move(self.Right_Widget.size().width()/2 - button0.size().width()/2,self.Right_Widget.size().height()-500)
        button0.setText("上一张")
        button0.setStyleSheet('font-size:22px;')
        button0.clicked.connect(self.PreviousImage)

        button1 = QPushButton(self.Right_Widget)
        button1.resize(200, 60)
        button1.move(self.Right_Widget.size().width()/2 - button0.size().width()/2,button0.pos().y() + button1.size().height() + 10)
        button1.setText("下一张")
        button1.setStyleSheet('font-size:22px;')
        button1.clicked.connect(self.NextImage)

        self.ShowImage(image)

    def ShowImage(self,img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.L_Width, self.L_Height-100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.Left_Widget.setPixmap(QPixmap.fromImage(img))

    def PreviousImage(self):
        pass

    def NextImage(self):
        pass

    # current text change function
    def ComboBoxChanged(self):
        if self.ComboBox.currentIndex() == 0:
            print(0)
        else:
            print(1)

    def UpdateAlbumList(self):
        pass