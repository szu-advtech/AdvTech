import os
import sys
import threading
import time

from PyQt5.Qt import *
from PyQt5.QtCore import QDateTime, QTimer, QSize
from PyQt5.QtWidgets import QWidget, QListWidget, QStackedWidget, QLabel, QAbstractItemView, QListWidgetItem

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)

import Album
import Display3D


class MainWindow(QWidget):
    def __init__(self, Width=1600, Height=900):
        super(MainWindow, self).__init__()
        # 创建类内变量
        self.Width = Width
        self.Height = Height
        # 主窗口size,之后计算子窗口大小
        self.resize(self.Width, self.Height)
        # 计算左侧List size
        L_Width = 200
        # L_Width = int(self.Width / 5)
        # L_Width = 200 if L_Width < 200 else L_Width
        # L_Width = 300 if L_Width > 300 else L_Width
        L_Height = int(self.Height / 10)
        L_Height = 100 if L_Height < 100 else L_Height
        L_Height = 200 if L_Height > 200 else L_Height
        self.L_Width, self.L_Height = L_Width, L_Height
        # 计算右侧窗口size
        self.R_Width = self.Width - self.L_Width
        self.R_Height = self.Height
        # 创建左右两个部件
        self.Left_Widget = QListWidget(self)
        self.Right_Widget = QStackedWidget(self)
        # self.setDesktopSize()
        # 左边小部件添加实时时间显示
        self.time = QDateTime.currentDateTime()  # 获取当前日期与时间 QDateTime.currentDateTime()
        self.time = self.time.toString("yyyy-MM-dd hh:mm:ss")
        self.TimeText = QLabel(self)
        self.TimeText.setText(self.time)
        self.TimeText.resize(self.L_Width, int(self.L_Height / 10))
        self.TimeText.setStyleSheet("font-size: 20px;")
        # 创建右边部件
        print(self.R_Width, self.R_Height)
        self.Right_Widget.resize(self.R_Width, self.R_Height)
        self.Right_Widget.move(self.L_Width, 0)
        # 右侧子窗口
        self.Display3d1 = Display3D.Display3D(self.R_Width, self.R_Height, mesh_path="./mesh/brandenburg.ply")
        self.Display3d2 = Display3D.Display3D(self.R_Width, self.R_Height, mesh_path="./mesh/lincoln_memorial.ply")
        self.Display3d2.SetTranslation_xyz(0, -2, 0)
        self.Display3d3 = Display3D.Display3D(self.R_Width, self.R_Height, mesh_path="./mesh/castle.ply")
        self.Display3d3.SetTranslation_xyz(0, 1, -4)
        self.Display3d4 = Display3D.Display3D(self.R_Width, self.R_Height, mesh_path="./mesh/L7.ply")
        self.Display3d4.SetTranslation_xyz(1, 2, -4)
        # 初始化主窗口
        self.Widget_init()  # 初始化左右部件
        self.setMouseTracking(False)  # 设置鼠标移动跟踪是否有效
        # 初始化窗口名
        self.setWindowTitle("NeuralRecon-W")
        # 新建一个QTimer对象
        self.timer = QTimer()
        self.timer.setInterval(1000)
        # 信号连接到槽
        self.timer.timeout.connect(self.ShowTime)
        threading.Thread(target=self.timer.start())

    def Widget_init(self):
        qss_path = os.path.join(os.path.join(os.path.abspath(dir_path + os.path.sep + "."), 'qss'),
                                'Left_Widget.qss')
        with open(qss_path, 'r') as f:  # 导入QListWidget的qss样式
            Widget_Style = f.read()
        self.Left_Widget.setStyleSheet(Widget_Style)
        self.Left_Widget.setSelectionMode(QAbstractItemView.SingleSelection)
        # '''加载界面ui'''
        self.Left_Widget.currentRowChanged.connect(self.Right_Widget.setCurrentIndex)  # list和右侧窗口的index对应绑定
        self.Left_Widget.setFrameShape(QListWidget.NoFrame)  # 去掉边框
        self.Left_Widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏滚动条
        self.Left_Widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 左侧子选项
        self.Left_Widget.item0 = QListWidgetItem('branden_burg', self.Left_Widget)  # 左侧选项的添加
        self.Left_Widget.item1 = QListWidgetItem('lincoln_memorial', self.Left_Widget)
        self.Left_Widget.item2 = QListWidgetItem('castle', self.Left_Widget)
        self.Left_Widget.item3 = QListWidgetItem('l7', self.Left_Widget)
        self.Left_Widget.item0.setTextAlignment(Qt.AlignCenter)  # 居中显示
        self.Left_Widget.item1.setTextAlignment(Qt.AlignCenter)  # 居中显示
        self.Left_Widget.item2.setTextAlignment(Qt.AlignCenter)  # 居中显示
        self.Left_Widget.item3.setTextAlignment(Qt.AlignCenter)  # 居中显示
        # 右边部分加入界面
        self.Right_Widget.addWidget(self.Display3d1)
        self.Right_Widget.addWidget(self.Display3d2)
        self.Right_Widget.addWidget(self.Display3d3)
        self.Right_Widget.addWidget(self.Display3d4)
        # 设置左边List，size
        self.Left_Widget.resize(self.L_Width, self.Height)
        self.Left_Widget.move(0, 0)
        self.Left_Widget.item0.setSizeHint(QSize(self.L_Width, self.L_Height))
        self.Left_Widget.item1.setSizeHint(QSize(self.L_Width, self.L_Height))
        self.Left_Widget.item2.setSizeHint(QSize(self.L_Width, self.L_Height))
        self.Left_Widget.item3.setSizeHint(QSize(self.L_Width, self.L_Height))
        self.TimeText.resize(self.L_Width, int(self.Height / 12))
        self.TimeText.move(5, self.Height - int(self.TimeText.height() - 10))

    def setDesktopSize(self):
        desktop = QDesktopWidget()
        self.Width = desktop.screenGeometry().width()
        self.Height = desktop.screenGeometry().height()

    def ShowTime(self):
        self.time = QDateTime.currentDateTime()  # 获取当前日期与时间 QDateTime.currentDateTime()
        self.time = self.time.toString("yyyy-MM-dd hh:mm:ss")
        self.TimeText.setText(self.time)
        time.sleep(0.5)

    def closeEvent(self, event):
        pass
        # self.OutdoorMonitoring.MonitorTab.closeEvent()
        # self.IntdoorMonitoring.MonitorTab.closeEvent()
        # self.NodeDataPage.TCPSer.stop()

    # 右边部分鼠标检测
    # def mousePressEvent(self, event):
    #     print("鼠标按下")
    #     print(QCursor.pos().x())
    #     print(QCursor.pos().y())
    #
    # def mouseReleaseEvent(self, event):
    #     print("鼠标释放")
    #     print(QCursor.pos().x())
    #     print(QCursor.pos().y())

    # def timerEvent(self, event):
