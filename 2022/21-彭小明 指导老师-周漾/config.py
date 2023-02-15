#这是win10下的设置
#meshlabserver_cmd = "MeshLab2022.02-windows.exe"
#这是在服务器上的Ubuntu下的设置
meshlabserver_cmd = "/tmp/pycharm_project_708/DRT-master/tmp/pycharm_project_708/DRT-master/MeshLab2022.02-linux.AppImage"
# if you are using a headless server, you may need to prepend `DISPLAY=:0`
# meshlabserver_cmd = "DISPLAY=:0 " + meshlabserver_cmd

# choose a directory to exchange temporary mesh file with meshlabserver
#tmp_path = "/dev/shm/DR/"
#这是win10下的设置
#tmp_path = "D:/paper_project/DRT-master/DRT-master/temp_mesh"
#这是在服务器上的Ubuntu下的设置
tmp_path="/tmp/pycharm_project_708/DRT-master/temp_mesh"

# path to hdf5 file and visual hull mesh
#data_path = "./data/Data_Redmi/"
#这是win10下的设置
#data_path="D:/paper_project/DRT-master/DRT-master/data/Data_Redmi/"
#这是在服务器上的Ubuntu下的设置
data_path="/tmp/pycharm_project_708/DRT-master/data/Data_Redmi/"
#result_path = "./result/"
#这是win10下的设置
#result_path="D:/paper_project/DRT-master/DRT-master/result/"
#这是在服务器上的Ubuntu下的设置
result_path="/tmp/pycharm_project_708/DRT-master/result/"

HyperParams = {
    # available model:
    # hand, mouse, monkey, horse, dog, rabbit, tiger, pig
    'name' :  'rabbit',   #'name' :  'pig',
    'IOR' : 1.4723,
    'Pass' : 20, # num of optimization stages,原本20
    'Iters' : 200, # in each stage,原本200

    # loss weight
    "ray_w" : 40,
    "sm_w": 0.08,
    # "sm_w": 0.02,
    "vh_w": 2e-3,

    # optimization parameters
    "momentum": 0.95,
    "start_lr": 0.1,
    "lr_decay": 0.5,
    "start_len": 10, # remesh target length
    "end_len": 1, # remesh target length
    # "end_len": 0.5, 
    'num_view': 72, # used for refraction loss  #注意在captured_data中要相应地修改num_view
                }