import numpy as np
import open3d as o3d



if __name__ == '__main__':
    PolyTech_S_path = "./pointcloud/PolyTech_S.ply"
    PolyTech_S_seg_path = "./pointcloud/PolyTech_S - Cloud.ply"
    # 载入点云
    # PolyTech_S = o3d.io.read_point_cloud(PolyTech_S_path)
    PolyTech_S_seg = o3d.io.read_point_cloud(PolyTech_S_seg_path)


    # 设置点云颜色
    # PolyTech_S.paint_uniform_color([1, 1, 1])
    PolyTech_S_seg.paint_uniform_color([1.0, 0.0, 0.0])

    # 计算包围盒
    aabb = PolyTech_S_seg.get_axis_aligned_bounding_box()
    aabb.color = (0,1,0)
    [center_x, center_y, center_z] = aabb.get_center()
    min_bound = np.asarray(aabb.get_min_bound())

    vector1 = np.array([center_x, center_y, center_z])
    vector2 = min_bound
    r = np.linalg.norm(vector1 - vector2)
    # 包围球
    # Radius = 2.818827 # 球半径
    Radius = 3.5  # 球半径
    Resolution = 10 # 球分辨率
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=Radius,
                                                          resolution=Resolution)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([1, 1, 1])
    mesh_sphere.translate((center_x+0.2, center_y-0.7, center_z), relative=True)
    points = np.asarray(mesh_sphere.vertices)
    lines = []
    for line in mesh_sphere.triangles:
        if [line[1],line[2]] not in lines:
            lines.append([line[1],line[2]])
    for i in range(4 * Resolution):
        line = mesh_sphere.triangles[i]
        if [line[0],line[1]] not in lines:
            lines.append([line[0],line[1]])
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    mesh_box = o3d.geometry.TriangleMesh.create_box(width=2*Radius,
                                                    height=2*Radius,
                                                    depth=2*Radius)
    mesh_box.translate((center_x+0.2 - Radius, center_y-0.7 - Radius, center_z - Radius), relative=True)

    box_points = np.asarray(mesh_box.vertices)
    bound_box = np.array([[0, 1],[0, 2],[0, 4],
                          [1, 3],[1, 5],[2, 3],
                          [2, 6],[4, 5],[4, 6],
                          [7, 5],[7, 6],[7, 3]])
    colors = [[0, 1, 0] for i in range(len(bound_box))]
    bound_box_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(box_points),
        lines=o3d.utility.Vector2iVector(bound_box),
    )
    bound_box_set.colors = o3d.utility.Vector3dVector(colors)
    print("origin: ",[center_x+0.2, center_y-0.7, center_z])
    print("Radius: ",Radius)
    print("eval_bbx: ",[[center_x+0.2 - Radius, center_y-0.7 - Radius, center_z - Radius],[center_x+0.2 + Radius, center_y-0.7 + Radius, center_z + Radius]])

    # 点云显示
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Open3d")
    # 设置窗口大小
    vis.create_window(width=1600, height=900)
    # 设置点云大小
    vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # 加入点云
    # vis.add_geometry(PolyTech_S)
    vis.add_geometry(PolyTech_S_seg)
    # vis.add_geometry(aabb)
    # vis.add_geometry(mesh_sphere)
    vis.add_geometry(line_set)
    vis.add_geometry(bound_box_set)

    #
    # o3d.visualization.draw_geometries([PolyTech_S,PolyTech_S_seg],
    #                                   width=1600,
    #                                   height=900)
    vis.run()
    vis.destroy_window()
