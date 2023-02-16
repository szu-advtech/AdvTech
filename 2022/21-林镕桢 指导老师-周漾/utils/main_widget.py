import sys
from PyQt6.QtCore import QSize 
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QComboBox, QLabel, QSpinBox, QFileDialog, QCheckBox)
from PyQt6.QtGui import QColor, QPixmap, QIcon
from utils.paint_board import PaintBoard
from utils.label_list import sub_list


class MainWidget(QWidget):
    def __init__(self, img_handler):
        super().__init__()
        
        self.img_handler = img_handler
        self.cur_feature = img_handler.get_feature("./origin.jpg")
        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
    
    def __InitData(self):
        self.__paintBoard = PaintBoard(self)
        self.__colorList = QColor.colorNames() 
        
    def __InitView(self):
        self.setFixedSize(1080,680)
        self.setWindowTitle("SPADE PaintBoard")
        
        main_layout = QVBoxLayout(self) 
        main_layout.setSpacing(10) 
    
        board = QHBoxLayout()
        board.setContentsMargins(10, 10, 10, 10) 
        board.addWidget(self.__paintBoard) 

        self.img_label = QLabel(self)
        self.update_img()
        board.addWidget(self.img_label)
        main_layout.addLayout(board)
        
        sub_layout = QHBoxLayout() 
        sub_layout.setContentsMargins(10, 10, 10, 10) 

        buttons = QVBoxLayout()

        click_box = QHBoxLayout() 
        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        
        click_box.addWidget(self.__cbtn_Eraser)

        self.__btn_infer = QPushButton("  使用参考")
        self.__btn_infer.setParent(self)
        self.__btn_infer.clicked.connect(self.on_infer_clicked)
        click_box.addWidget(self.__btn_infer)

        buttons.addLayout(click_box)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self) #设置父对象为本界面
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) 
        buttons.addWidget(self.__btn_Clear)
        
        # self.__btn_Quit = QPushButton("退出")
        # self.__btn_Quit.setParent(self) #设置父对象为本界面
        # self.__btn_Quit.clicked.connect(self.Quit)
        # sub_layout.addWidget(self.__btn_Quit)
        
        
        file_opt = QHBoxLayout()

        self.__btn_Open = QPushButton("打开语义")
        self.__btn_Open.setParent(self)
        self.__btn_Open.clicked.connect(self.on_btn_Open_Clicked)
        file_opt.addWidget(self.__btn_Open)
        
        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        file_opt.addWidget(self.__btn_Save)

        buttons.addLayout(file_opt)
        
        sub_layout.addLayout(buttons)
        

        choose = QVBoxLayout()
        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        choose.addWidget(self.__label_penColor)
        
        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor) #用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange) #关联下拉列表的当前索引变更信号与函数on_PenColorChange
        choose.addWidget(self.__comboBox_penColor)

        thickness = QHBoxLayout()

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)

        thickness.addWidget(self.__label_penThickness)
        
        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(100)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(30) #默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2) #最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)#关联spinBox值变化信号和函数on_PenThicknessChange
        thickness.addWidget(self.__spinBox_penThickness)

        choose.addLayout(thickness)
        sub_layout.addLayout(choose)
        main_layout.addLayout(sub_layout) #将子布局加入主布局


    def update_img(self):
        self.img_handler.gen_img("./temp.png", self.cur_feature)
        # self.img = QPixmap("./temp.png")
        self.img_label.setPixmap(QPixmap("./gen_img.png").scaled(512,512))


    def __fillColorList(self, comboBox):

        index = 0
        self.ci_to_l = []
        for label in sub_list.keys():
            color = self.__colorList[index]
            self.ci_to_l.append(sub_list[label])
            index += 1
            pix = QPixmap(20,20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), label)
            comboBox.setIconSize(QSize(20,20))
            # comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(0)
        
    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        cur_label = self.ci_to_l[color_index] - 1 
        self.__paintBoard.ChangePenColor(QColor(color_str), QColor(cur_label, color_index, 0))
        

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)
    
    def on_btn_Open_Clicked(self):
        openPath = QFileDialog.getOpenFileName(self, 'Choose your seg map', 'E:/Datasets/cocostuff/val_label', '*.png')
        print(openPath)
        if openPath[0] == "":
            print("Open cancel")
            return
        self.img_handler.gen_from_seg(openPath[0], self.cur_feature)
        self.img_label.setPixmap(QPixmap("./gen_img.png").scaled(512,512))

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        
    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True #进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False #退出橡皮擦模式


    def on_infer_clicked(self):
        openPath = QFileDialog.getOpenFileName(self, 'Choose your style map', 'E:/Datasets/cocostuff/val_img', '*.jpg')
        print(openPath)
        if openPath[0] == "":
            print("Open cancel")
            return
        self.cur_feature = self.img_handler.get_feature(openPath[0])
        self.update_img()
        # self.img_label.setPixmap(QPixmap("./gen_img.png").scaled(512,512))
        
        
    def Quit(self):
        self.close()

    
