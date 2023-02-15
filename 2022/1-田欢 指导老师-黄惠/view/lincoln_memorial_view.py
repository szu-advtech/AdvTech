import numpy as np
import open3d as o3d
import win32api

if __name__ == '__main__':
    mesh_path = "../mesh/lincoln_memorial.ply"
    # 载入点云
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # 计算包围盒
    aabb = mesh.get_axis_aligned_bounding_box()
    [center_x, center_y, center_z] = aabb.get_center()
    # print([center_x, center_y, center_z])
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi, np.pi))#绕y轴旋转-180°
    mesh.rotate(R, center=(center_x, center_y, center_z))
    # mesh.translate((0, -2, 1))
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Open3d")
    # 设置窗口大小
    vis.create_window(width=1200, height=900)
    # # 设置颜色背景为黑色
    vis.get_render_option().background_color = np.asarray([0,0,0])
    vis.add_geometry(mesh)
    # 可视化
    while True:
        R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 90, 0))
        mesh.rotate(R, center=(center_x, center_y, center_z))
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        esc = win32api.GetKeyState(27)
        if esc < 0:
            exit()
            vis.destroy_window()

