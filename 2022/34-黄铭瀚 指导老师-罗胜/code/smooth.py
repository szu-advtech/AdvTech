import bpy
import bmesh

names = [o.name for o in bpy.context.selected_objects]

max_len = 0
max_name = ''
for name in names:

     # select the object
    obj = bpy.data.objects[name]
    vertices_len = len(obj.data.vertices)
    if vertices_len > max_len:
        max_len = vertices_len
        max_name = name

bpy.ops.object.select_all(action='DESELECT')

for name in names:
    if name != max_name:
        obj = bpy.data.objects[name]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        vertex_group = obj.vertex_groups.new( name = 'boundary' )
        for vert in bm.verts:
            if vert.is_boundary:
                vertex_group.add( [vert.index], 1, 'ADD' )
        print('complete')
        bpy.ops.object.modifier_add(type='LAPLACIANSMOOTH')
        bpy.context.object.modifiers["Laplacian Smooth"].vertex_group = "boundary"
        bpy.context.object.modifiers["Laplacian Smooth"].iterations = 100
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Laplacian Smooth")
        obj.select_set(False)
    
for name in names:
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    

    
    