import numpy as np
from statistics import mean
import open3d as o3d
from multiprocessing import Process, Queue
import queue

def get_normal(q, input_normals, pcd_tree, vertices_normal_part, list_of_vertices):
    vertices_num = len(list_of_vertices)
    for vid in range(vertices_num):
        pid = list_of_vertices[vid][0]
        idx = list_of_vertices[vid][1]
        vertice = list_of_vertices[vid][2]

        [k, idces, _] = pcd_tree.search_knn_vector_3d(vertice, 10)
        neighbor_normals = input_normals[idces]
        mean_normal = neighbor_normals.mean(axis=0)
        vertices_normal_part[idx] = mean_normal
            
    q.put([1, pid, vertices_normal_part])


def mesh_reorient(dataset_pc, vertices, triangles): 
    if dataset_pc.by_voxel_size:
        vertices = vertices * dataset_pc.voxel_size + dataset_pc.pc_min
    else:
        vertices = vertices  / (dataset_pc.output_grid_size * dataset_pc.block_num_per_dim) * dataset_pc.pc_scale + dataset_pc.pc_min
    
    maicity = False
    if maicity:
        vertices[:, 2] = -vertices[:, 2]
    
    reorient_flag = True 
    if reorient_flag:
        point_cloud_data = dataset_pc.input_point        
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_data) 
        input_normals = np.asarray(point_cloud_data.normals)
        
        #prepare list of vertices
        even_distribution = [16]
        this_machine_id = 0
        num_of_process = 0
        P_start = 0
        P_end = 0
        for i in range(len(even_distribution)):
            num_of_process += even_distribution[i]
            if i < this_machine_id:
                P_start += even_distribution[i]
            if i <= this_machine_id:
                P_end += even_distribution[i]
        print(this_machine_id, P_start, P_end)
        
        list_of_list_of_vertices = []
        for i in range(num_of_process):
            list_of_list_of_vertices.append([])

        vertices_num = len(vertices)
        for idx in range(vertices_num):
            process_id = idx % num_of_process
            list_of_list_of_vertices[process_id].append([process_id, idx, vertices[idx]])
        
        #map processes
        q = Queue()
        workers = []
        vertices_normal_part = np.zeros([vertices_num, 3]).astype(np.float32)

        for i in range(P_start,P_end):
            list_of_vertices = list_of_list_of_vertices[i]
            workers.append(Process(target=get_normal, args = (q, input_normals, pcd_tree, vertices_normal_part, list_of_vertices)))

        for p in workers:
            p.start()

        vertices_normal_q = np.zeros([num_of_process, vertices_num, 3]).astype(np.float32)
        for i in range(num_of_process):
            _, pid, vertices_normal_part = q.get()
            vertices_normal_q[pid] = vertices_normal_part

        vertices_normal = vertices_normal_q.sum(axis=0)
        
        print("finished knn")
        
        triangles_num = len(triangles)
        triangles_np = np.array(triangles)
        triangles_np = triangles_np.flatten()
        triangles_normal = vertices_normal[triangles_np]
        triangles_normal = triangles_normal.reshape(triangles_num, 3, 3)
        mean_n = triangles_normal.mean(axis=1)
        
        vertices_triangles = vertices[triangles_np]
        vertices_triangles = vertices_triangles.reshape(triangles_num, 3, 3)
        ab = vertices_triangles[:,1] - vertices_triangles[:,0]
        bc = vertices_triangles[:,2] - vertices_triangles[:,1]
        face_n = np.cross(ab, bc)
        
        res = (face_n * mean_n).sum(axis=1)
        idx = np.argwhere(res < 0)
        tmp = triangles[idx,1]
        triangles[idx,1] = triangles[idx,2]
        triangles[idx,2] = tmp
        
        print("finished reoriention")
             
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    if reorient_flag:
        output_mesh_path = "./samples/" + dataset_pc.file_name +"_reorient_mesh.ply"
    else:
        output_mesh_path = "./samples/" + dataset_pc.file_name +"_undc_mesh.ply"
        
    o3d.io.write_triangle_mesh(output_mesh_path, mesh, write_ascii=True)
    return mesh
