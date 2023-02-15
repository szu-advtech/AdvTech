import open3d as o3d
import numpy as np
import copy

path = './data/modelnetply/chair/train/'

def demo_crop_geometry():
    print("手动几何裁剪演示")
    print(
        "1) 按两次“Y”以将几何体与Y轴的负方向对齐"
    )
    print("2) 按“K”锁定屏幕并切换到选择模式")
    print("3) 拖动以选择矩形,")
    print("   或者使用ctrl+左键单击进行多边形选择")
    print("4) 按“C”获取选定的几何图形并保存")
    print("5) 按“F”切换到自由视图模式")
    # 加载点云
    pcd = o3d.io.read_point_cloud(path+'chair_0005.ply')
    # 可视化几何体供用户交互
    o3d.visualization.draw_geometries_with_editing([pcd])


def visual_selection(source):
    """绘制配准结果"""
    source_temp = copy.deepcopy(source)
    # target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([source_temp, target_temp])
    o3d.visualization.draw_geometries([source_temp])


def pick_points(pcd):
    print("")
    print(
        "1) 请使用至少选择三个对应关系 [shift + 左击]"
    )
    print("   按 [shift + 右击] 撤销拾取的点")
    print("2) 拾取点后，按“Q”关闭窗口")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    # 激活窗口。此函数将阻止当前线程，直到窗口关闭。
    vis.run()  # 等待用户拾取点
    vis.destroy_window()
    print("")
    return vis.get_picked_points()
 
 
def demo_manual():
    # 加载点云
    source = o3d.io.read_point_cloud(path+'chair_0005.ply')
    # target = o3d.io.read_point_cloud(path)
    # visual_selection(source)
 
    # 从两点云中拾取点并建立对应关系
    picked_id_source = pick_points(source)
    # picked_id_target = pick_points(target)
    print(picked_id_source)

if __name__ == '__main__':
    # demo_crop_geometry()
    demo_manual()