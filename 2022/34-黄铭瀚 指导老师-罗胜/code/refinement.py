import bpy
import numpy as np
import bmesh


def get_index_of_selected_faces():
    mode = bpy.context.active_object.mode
    # Keep track of previous mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Go into object mode to update the selected vertices
    obj = bpy.context.object
    # Get the currently select object
    sel = np.zeros(len(obj.data.polygons), dtype=np.bool)
    # Create a numpy array with empty values for each vertex
    obj.data.polygons.foreach_get('select', sel)
    bpy.ops.object.mode_set(mode=mode)
    return np.where(sel==True)[0]

obj = bpy.context.active_object
face_group = get_index_of_selected_faces()
#print(face_group)
mesh = obj.data
bpy.ops.object.mode_set(mode='OBJECT')
for index in face_group:
    mesh.polygons[index].material_index = 0
    
bpy.ops.object.mode_set(mode='EDIT')

 