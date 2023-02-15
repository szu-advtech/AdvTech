import open3d as o3d
import os
# import sys
# sys.path.append("..")
# import dataloader.pcd_loader

shapenet_path = './data/animal.obj'
point_cloud_path = {'inhouse':'./data/pointcloud/fragment.ply',
            }
shape_net = {'rocket':'./data/shapenet/rocket.obj',
            'M4A1':'./data/shapenet/M4A1.obj',
            'printer':'./data/shapenet/printer.obj',
            'cup':'./data/shapenet/cup.obj',
            'bat_motor':'./data/shapenet/bat_motor.obj',
            'lamp':'./data/shapenet/lamp.obj',
            'guitar':'./data/shapenet/guitar.obj',
            'car':'./data/shapenet/car.obj',
            'bus':'./data/shapenet/bus.obj',
            'bench':'./data/shapenet/bench.obj',
            'animal':'./data/shapenet/animal.obj',
            'AK47':'./data/shapenet/AK47.obj',
            }

ply_path = './data/modelnetply/desk/train/desk_0011.ply'
ball_path = './template/balls/2048.xyz'

# Load a triangle mesh or ply point cloud, print it, and render it

def draw_pcd(file_type,mesh_or_pcd,path,point_num):
    # armadillo_data = o3d.data.ArmadilloMesh()
    # pcd = o3d.io.read_triangle_mesh(armadillo_data.path).sample_points_poisson_disk(point_num)
    
    #convert mesh to a point cloud and estimate dimensions.
    if mesh_or_pcd == 1:
        if file_type == 1:
            pcd = o3d.io.read_triangle_mesh(path)
        else:
            pcd = o3d.io.read_point_cloud(path)
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                pcd,densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=9)
        print("Displaying mesh ...")
        print(pcd)
        pcd.compute_vertex_normals()
        o3d.visualization.draw_geometries([pcd],mesh_show_back_face = True,window_name = "mesh")
    else:
        if file_type == 1:
            pcd = o3d.io.read_triangle_mesh(path).sample_points_poisson_disk(point_num)
        else:
            pcd = o3d.io.read_point_cloud(path)
        print("Displaying point cloud ...")
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw([pcd],point_size = 5)

#imput_file:     1.mesh,  2.point_cloud
#visualization:  1.mesh,  2.point_cloud
if __name__ == "__main__":
    path = os.path.join('D:/sp-gan_1129',shape_net['AK47'])
    draw_pcd(1,1,ply_path,2048)

    # train_dir = 'airplane/'+train_dir+'/airplane_0001.off'
    # path = os.path.join(model_net,train_dir)
    # path = '../data/ModelNet40/airplane/train/airplane_0001.off'
    # draw_pcd(1,2,path,2048)


