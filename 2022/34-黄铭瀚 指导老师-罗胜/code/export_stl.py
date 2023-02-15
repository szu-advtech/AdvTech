import bpy
import os
import numpy as np
#import sys
#sys.path.append(r"E:\Desktop\blender")
#bpy.ops.object.mode_set(mode='EDIT')
#from concave_manual import get_keypoint_lable
# get the current path and make a new folder for the exported meshes
names = [o.name for o in bpy.context.selected_objects]
destination_path = r'E:\Desktop\1213'
name = names[0]
if '.' in name:
    name = name[0:len(name) - 4]    
file_path = os.path.join(destination_path, name)

# if path doesn't exist, create, else skip that step
if not os.path.exists(file_path):
    os.makedirs(file_path)
npy_path = os.path.join(file_path, name +'_1.npy')
keypoint_lable = np.load(npy_path,allow_pickle=True)
#obj = bpy.context.active_object
#flag = False
#for vertex in obj.data.vertices:
#    for lable in keypoint_lable:
#        lable_item = list(lable.items())[0]
#        key, value = lable_item
#        #print(key, np.array(value))
#        a = np.where((np.array(value) == np.array(vertex.co)).all(1))[0]
#        if len(a):
#            print(a)
#            print(key, value)
#            flag =True
#            break
#    if flag:
#        break
#        #print(np.array(vertex.co))
#####
# deselect all
bpy.ops.object.select_all(action='DESELECT')
selected_index = []
# iterate over named objects
for name in names:

     # select the object
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    teeth_index = 0
    flag = False
    for vertex in obj.data.vertices:
        for lable in keypoint_lable:
            lable_item = list(lable.items())[0]
            key, value = lable_item
            #print(key, np.array(value))
            a = np.where((np.array(value) == np.array(vertex.co)).all(1))[0]
            if len(a):
                teeth_index = key
                #print(a)
                print(key, value)
                flag =True
                break
        if flag:
            break
    if '.' in name:
        name = name[0:len(name) - 4] 
    stl_path = os.path.join(file_path, name +'_' + str(teeth_index) +'.stl')
    #ply_path = os.path.join(file_path, name + '_' + str(teeth_index) + '.ply')
    #bpy.ops.export_mesh.ply(filepath=ply_path)
    if len(obj.data.vertices) > 100:
        bpy.ops.export_mesh.stl(filepath=stl_path,use_selection=True)
    obj.select_set(False)

for name in names:
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)