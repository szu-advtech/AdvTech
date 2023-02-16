import threading

import numpy as np
from PyQt5.Qt import *
from PyQt5.QtGui import QColor, QMatrix4x4
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.Qt3DExtras import Qt3DWindow, QTorusMesh, QPhongMaterial, QSphereMesh, QAbstractCameraController
from PyQt5.Qt3DInput import QInputAspect
from PyQt5.Qt3DCore import QEntity, QTransform
from PyQt5.Qt3DRender import QPointLight, QMesh
from PyQt5.Qt3DExtras import QFirstPersonCameraController

import open3d as o3d


class Display3D(QWidget):
    def __init__(self, Width=1500, Height=800, mesh_path="./mesh/lincoln_memorial.ply"):
        super(Display3D, self).__init__()
        self.Width = Width
        self.Height = Height
        # 设置主窗口size
        self.resize(self.Width, self.Height)

        self.Window3D = Qt3DWindow()
        self.Window3D.resize(self.size().width(), self.size().height())
        self.Window3D.defaultFrameGraph().setClearColor(QColor(128, 138, 135))

        container = QWidget.createWindowContainer(self.Window3D)
        container.resize(self.size().width(), self.size().height())

        # 根框架
        self.RootEntity = QEntity()
        self.Material = QPhongMaterial()
        self.Material.setAmbient(QColor(0,0,0))
        # self.Material.setDiffuse(QColor(255,120,100))


        HBoxLayout = QHBoxLayout(self)
        VBoxLayout = QVBoxLayout()
        VBoxLayout.setAlignment(Qt.AlignTop)
        HBoxLayout.addWidget(container, 1)
        HBoxLayout.addLayout(VBoxLayout)

        self.InPut3D = QInputAspect()
        self.Window3D.registerAspect(self.InPut3D)

        # Camera
        self.Camera = self.Window3D.camera()
        # self.Camera.setPosition(QVector3D(0, 5, 10.0))

        # self.Camera.setProjectionMatrix(QMatrix4x4(45.0, 16.0 / 9.0, 0.1, 1000.0))
        self.Camera.setPosition(QVector3D(0, 0, 10))
        self.Camera.setUpVector(QVector3D(0, 1, 0))
        self.Camera.setViewCenter(QVector3D(0, 0, 0))
        # self.Camera.setViewCenter(QVector3D(0, 2, 0))

        # 球体形状数据
        sphereMesh = QSphereMesh()
        sphereMesh.setRings(20)
        sphereMesh.setSlices(20)
        sphereMesh.setRadius(2)

        # mesh = o3d.io.read_triangle_mesh(meshpath)
        # mesh.translate((0, 0, 0), relative=True)
        # o3d.io.write_triangle_mesh(meshpath,mesh)
        self.Mesh = QMesh()
        self.Url = QUrl().fromLocalFile(mesh_path)
        self.Mesh.setSource(self.Url)

        self.toursEntity = QEntity(self.RootEntity)
        # For Camera Controls
        self.CamController = QFirstPersonCameraController(self.RootEntity).setCamera(self.Camera)
        # self.CamController = QAbstractCameraController(self.RootEntity).setCamera(self.Camera)

        self.toursTransform = QTransform()
        self.toursTransform.setRotationX(180)
        self.toursTransform.setRotationY(2)
        self.toursTransform.setRotationZ(0)
        self.toursTransform.setTranslation(QVector3D(0, 0, 0))
        # self.toursTransform.setScale3D(QVector3D(10,10,10))

        self.PointLight = QPointLight(self.RootEntity)
        self.PointLight.setColor(QColor(255, 255, 255))
        self.PointLight.setIntensity(1.1)

        self.toursEntity.addComponent(self.Mesh)
        # self.toursEntity.addComponent(sphereMesh)
        self.toursEntity.addComponent(self.PointLight)
        self.toursEntity.addComponent(self.toursTransform)
        self.toursEntity.addComponent(self.Material)

        self.Window3D.setRootEntity(self.RootEntity)

        # 新建一个QTimer对象
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.angle = 0
        self.flage = False
        # 信号连接到槽
        self.timer.timeout.connect(self.RotateMesh)
        # self.timer.start()
        # threading.Thread(target=self.timer.start())

    def RotateMesh(self):
        hz = 2
        # self.toursTransform.setRotationY(2 + self.angle)

        self.toursEntity.addComponent(self.toursTransform)
        self.Window3D.setRootEntity(self.RootEntity)
        if self.flage:
            self.angle -= hz
        else:
            self.angle += hz
        if self.angle > 90:
            self.flage = True
        elif self.angle < -90:
            self.flage = False

    def SetTranslation_xyz(self, x, y, z):
        self.toursTransform.setTranslation(QVector3D(x, y, z))
        self.toursEntity.addComponent(self.toursTransform)

    def SetRotationX(self, R):
        self.toursTransform.setRotationX(R)
        self.toursEntity.addComponent(self.toursTransform)

    def SetRotationY(self, R):
        self.toursTransform.setRotationX(R)
        self.toursEntity.addComponent(self.toursTransform)

    def SetRotationZ(self, R):
        self.toursTransform.setRotationX(R)
        self.toursEntity.addComponent(self.toursTransform)
