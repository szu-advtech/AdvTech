import open3d as o3d

pcd = o3d.io.read_point_cloud(".\outputs/mvsnet075_l3.ply")
print(pcd) #打印简单的信息：TriangleMesh with 1440 points and 2880 triangles.
#voxel_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh(pcd,voxel_size)

# 写入(这里是复制)一份新数据
#o3d.io.write_triangle_mesh("copy_of_mvsnet15_l3.ply", mesh)

#显示
o3d.visualization.draw_geometries([pcd])