# 注意事项

本系统参考的原始论文为TANDEM（https://github.com/tum-vision/tandem），这是一个利用单目透视视频进行三维重建的工作，使用DSO作为前端视觉里程计估计相机位姿并挑选关键帧，将关键帧传入CVA-MVSNet模块估计对应的深度值，最后将RGBD图像传入TSDF Fusion模块完成三维重建。本系统仿照TANDEM系统框架，同样设置了SLAM、深度估计和点云融合模块，但由于输入变为了全景图，为应对其特殊的投影格式和畸变，每个模块都需要另外寻找或编写适用于全景图的算法。

具体来说，本系统以OpenVSLAM（https://github.com/zm0612/openvslam-comments）、HoHoNet（https://github.com/sunset1995/HoHoNet）和Potree（https://github.com/potree/potree）的源码为基础，实现一个真实场景点云在线生成与渲染的系统。