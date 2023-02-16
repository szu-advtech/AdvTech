import os
import numpy as np
import argparse
from multiprocessing import Process, Queue
import queue

# normalize the meshes and remove empty folders.
parser = argparse.ArgumentParser()
parser.add_argument("--share_id", action="store", dest="share_id", type=int, help="id of the share [0]")
parser.add_argument("--share_total", action="store", dest="share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

target_dir = "./objs/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

start = int(share_id*len(obj_names)/share_total)
end = int((share_id+1)*len(obj_names)/share_total)
obj_names = obj_names[start:end]

def load_obj(dire):
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            vertices.append([x,y,z])
        if line[0] == 'f':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale # 先将整体平移到零点位置，然后进行缩放
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    return vertices, triangles

def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()

def simplify_obj(q, name_list):
    name_num = len(name_list)
    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        this_subdir_name = name_list[nid][2]

        # this_subdir_name = target_dir + obj_names[i]
        sub_names = os.listdir(this_subdir_name)
        if len(sub_names)==0: # 判断该文件夹是否为空
            command = "rm -r "+this_subdir_name
            os.system(command)
        else:
            this_name = this_subdir_name+"/"+sub_names[0]
            out_name = this_subdir_name+"/model.obj"
            if not os.path.exists(out_name):
                print(pid, idx, this_name)

                v,t = load_obj(this_name)
                write_obj(out_name, v,t)
            else:
                print(pid, idx, out_name +" is exist")

        q.put([1,pid,idx])



# 下面就开始慢慢使用并行运算了
obj_names_len = len(obj_names)

#prepare list of names
even_distribution = [16]
this_machine_id = 0
num_of_process = 0
P_start = 0
P_end = 0
for i in range(len(even_distribution)):
    num_of_process += even_distribution[i]
    if i<this_machine_id:
        P_start += even_distribution[i]
    if i<=this_machine_id:
        P_end += even_distribution[i]
print(this_machine_id, P_start, P_end)

list_of_list_of_names = []
for i in range(num_of_process):
    list_of_list_of_names.append([])
for idx in range(obj_names_len):
    process_id = idx%num_of_process
    in_name = target_dir + obj_names[idx]

    list_of_list_of_names[process_id].append([process_id, idx, in_name])


#map processes
q = Queue()
workers = []
for i in range(P_start,P_end):
    list_of_names = list_of_list_of_names[i]
    workers.append(Process(target=simplify_obj, args = (q, list_of_names)))

for p in workers:
    p.start()


counter = 0
while True:
    item_flag = True
    try:
        success_flag, pid, idx = q.get(True, 1.0)
    except queue.Empty:
        item_flag = False
    
    if item_flag:
        #process result
        counter += success_flag

    allExited = True
    for p in workers:
        if p.exitcode is None:
            allExited = False
            break
    if allExited and q.empty():
        break


print("finished")
print("returned", counter,"/",obj_names_len)