from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QPainter, QPaintEvent, QMouseEvent, QPen, QColor
from PyQt6.QtCore import Qt, QPoint, QSize

class PaintBoard(QWidget):


    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.parent = Parent

        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
        
    def __InitData(self):
        
        self.__size = QSize(512,512)
        
        #新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(QSize(512,512))
        self.__board.fill(QColor(255, 255, 255)) #用白色填充画板
        self.__hidden_board = QPixmap(QSize(512,512))
        self.__hidden_board.fill(QColor(255, 0, 255)) #用白色填充画板
        self.__hidden_board.toImage().save("temp.png")
        
        
        self.__IsEmpty = True #默认为空画板 
        self.EraserMode = False #默认为禁用橡皮擦模式
        
        self.__lastPos = QPoint(0,0)#上一次鼠标位置
        self.__currentPos = QPoint(0,0)#当前的鼠标位置
        
        self.__painter = QPainter()#新建绘图工具
        # self.__painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.__painter_1 = QPainter()#新建绘图工具
        # self.__painter_1.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        self.__thickness = 30       #默认画笔粗细为10px
        self.__colorList = QColor.colorNames() #获取颜色列表
        self.__penColor = QColor(self.__colorList[0])#设置默认画笔颜色为黑色
        self.__penColor_1 = QColor(1,0,0)#设置默认画笔颜色为黑色
        
     
    def __InitView(self):
        #设置界面的尺寸为__size
        self.setFixedSize(self.__size)
        
    def Clear(self):
        #清空画板
        self.__board.fill(QColor(255, 255, 255))
        self.__hidden_board.fill(QColor(255, 0, 255))
        self.update()
        self.__IsEmpty = True
        
    def ChangePenColor(self, color, color_1):
        #改变画笔颜色
        self.__penColor = color
        self.__penColor_1 = color_1
        
    def ChangePenThickness(self, thickness=10):
        #改变画笔粗细
        self.__thickness = thickness
        
    def IsEmpty(self):
        #返回画板是否为空
        return self.__IsEmpty
    
    def GetContentAsQImage(self):
        #获取画板内容（返回QImage）
        image = self.__hidden_board.toImage()
        return image
        
    def paintEvent(self, paintEvent):
        self.__painter_1.begin(self)
        self.__painter_1.drawPixmap(0,0,self.__hidden_board)
        self.__painter_1.end()
        self.__painter.begin(self)
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()
        
        
    def mousePressEvent(self, mouseEvent):
        #鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos =  mouseEvent.pos()
        self.__lastPos = self.__currentPos
        
        
    def mouseMoveEvent(self, mouseEvent):
        #鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos =  mouseEvent.pos()

        #画线    
        

        self.__painter_1.begin(self.__hidden_board)
        if self.EraserMode == False:
            self.__painter_1.setPen(QPen(self.__penColor_1,self.__thickness)) 
        else:
            self.__painter_1.setPen(QPen(QColor(255, 0, 255),self.__thickness))
        self.__painter_1.drawLine(self.__lastPos, self.__currentPos)
        self.__painter_1.end()


        self.__painter.begin(self.__board)
        if self.EraserMode == False:
            self.__painter.setPen(QPen(self.__penColor,self.__thickness)) #设置画笔颜色，粗细
        else:
            self.__painter.setPen(QPen(QColor(255, 255, 255),self.__thickness))
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()

        self.__lastPos = self.__currentPos
                
        self.update() #更新显示

        self.__hidden_board.toImage().save("temp.png")
        
        
    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False #画板不再为空
        self.parent.update_img()


# class SubBoard
