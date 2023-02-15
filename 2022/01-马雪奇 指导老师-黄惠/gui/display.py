import sys

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow, MultiPlotter

class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame·
        self.frame = QtWidgets.QFrame()
        # create the vertical layout
        h_layout = QtWidgets.QHBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame, shape=(1, 2))
        self.plotter.show_axes_all()
        # pv.set_plot_theme("paraview")
        

        h_layout.addWidget(self.plotter.interactor)
        
        self.signal_close.connect(self.plotter.close)
        
        # self.signal_close.connect(self.multi_plotter[0,0].close)

        self.frame.setLayout(h_layout) # 将 layout 放在 frame 中
        self.setCentralWidget(self.frame) # 将 frame 放置在窗口的中间

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')        
        exitButton = QtWidgets.QAction('Exit', self) # 创建一个button 命名为 “Exit”
        exitButton.setShortcut('Ctrl+Q') # 添加快捷键
        exitButton.triggered.connect(self.close) # 设置 triggered 事件为 close
        fileMenu.addAction(exitButton) # 将该 exitButton 按钮添加到 fileMenu 里面

        # allow adding a point cloud
        pointCloudMenu = mainMenu.addMenu('Point Cloud')
        self.load_point_cloud_action = QtWidgets.QAction('load Point Cloud', self) # 好奇，这里为什么要用 self 呢？而上面就直接实例化一个对象呢？
        self.load_point_cloud_action.triggered.connect(self.load_point_cloud)
        pointCloudMenu.addAction(self.load_point_cloud_action)
        
        # allow adding a mesh
        meshMenu = mainMenu.addMenu('Mesh')        
        load_mesh_action = QtWidgets.QAction('load Mesh', self)
        load_mesh_action.triggered.connect(self.load_mesh)
        meshMenu.addAction(load_mesh_action)
        
        # allow linking two views
        cameraMenu = mainMenu.addMenu('Camera')
        link_views_action = QtWidgets.QAction('link views', self)
        link_views_action.triggered.connect(self.link_views)
        cameraMenu.addAction(link_views_action)
        
        unlink_views_action = QtWidgets.QAction('unlink views', self)
        unlink_views_action.triggered.connect(self.unlink_views)
        cameraMenu.addAction(unlink_views_action)

        if show:
            self.show()

    def load_point_cloud(self):
        point_cloud = pv.read('data/L7_60M_with_normal_combine.ply')
        # point_cloud = pv.read('data/ConvexHullPointCloud.ply')
        point_num=point_cloud.n_points

        self.plotter.subplot(0, 0)

        self.plotter.add_points(point_cloud, 
                                render_points_as_spheres=True, 
                                # color=[1.0, 1.0, 1.0],
                                color='#00FF00',
                                reset_camera=True, # 在 load mesh 的时候，重置一下 camera
                                label='point cloud',
                                point_size=1.0
                                )
        
        legend_entries = []
        legend_entries.append(['points: %d' % (point_num), 'w'])

        self.plotter.add_legend(
            # labels=['kk'],
            # bcolor='w'
            # size=(0.1, 0.05)
            legend_entries
            )
        
    def load_mesh(self):
        # sphere = pv.Sphere()
        mesh = pv.read('data/L7_60M_reorientation_raw_size.obj')
        # mesh = pv.read('data/ConvexHull_raw_size.obj')
        
        vertex_num = mesh.n_points
        face_num = mesh.n_faces
        self.plotter.subplot(0, 1)
        # self.plotter.show_axes()
        self.plotter.add_mesh(mesh, 
                                # show_edges=True,
                                lighting=True,
                                color=[1.0, 1.0, 1.0],
                                # opacity=0.7,
                                reset_camera=True,
                                smooth_shading=False,
                                diffuse=1.0,
                                pbr=False,
                                cmap='viridis_r',
                                label='mesh'                    
                                ) # 将 sphere 放置在 plotter 中
        
        floor = pv.Plane(center=(*mesh.center[:2], mesh.bounds[-2]-10), i_size=300, j_size=300)
        self.plotter.add_mesh(floor, color='gray')
        
        legend_entries = []
        legend_entries.append(['vertices: %d' % (vertex_num), 'w'])
        legend_entries.append(['faces: %d' % (face_num), 'w'])
        
        
        self.plotter.add_legend(
            # bcolor='w'            
            legend_entries
            # size=(0.1, 0.1)
            )

    def link_views(self):        
        # reset camera position
        self.plotter.reset_camera()
        
        self.plotter.link_views()
        
    def unlink_views(self):
        self.plotter.unlink_views()
        
    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    
    sys.exit(app.exec_())