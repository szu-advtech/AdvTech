import bpy
import numpy as np
from mathutils import kdtree
import bmesh
#import scipy
import scipy.sparse
import scipy.sparse.linalg

def concave_list(obj, num):
    #obj = bpy.context.active_object
    mesh = obj.data
    size = len(mesh.vertices)
    kd = kdtree.KDTree(size)
    theta = 0.01
    concave = [False] * size
    for index, vertice in enumerate(mesh.vertices):
        kd.insert(vertice.co, index)
    kd.balance()
    for index, vertice in enumerate(mesh.vertices):
        co_find = vertice.co
        co_find_norm = vertice.normal
        for (co, id, _) in kd.find_n(co_find, num):
            co_norm = mesh.vertices[id].normal
            dif_co = co_find - co
            dif_co = np.array(dif_co)
            dif_co_unit = dif_co / np.linalg.norm(dif_co)
            dif_norm = co_norm - co_find_norm
            # normals have normalized
            #dif_norm = np.array(dif_norm)
            #dif_norm_unit = dif_norm / np.linalg.norm(dif_norm)
            #res = np.dot(dif_co_unit, dif_norm_unit)
            res = np.dot(dif_co_unit, dif_norm)
            if(res > theta):
                concave[index] = True
                break
    return concave

def vertex_curvature_along_edge( vert, edge ):
    other = edge.other_vert( vert )
    normal_diff = other.normal - vert.normal
    vert_diff = other.co - vert.co
    return normal_diff.dot( vert_diff ) / vert_diff.length_squared

# AVG method: return the average of all adges curvatures
# A face that is inside is count twice in order to have the symetric one
# if the vertex in on the border of the figure
# Border edges are count once
def vertex_curvature_avg( vert ):
    result = 0.0
    count = 0
    for edge in vert.link_edges:
        faces = len(edge.link_faces) # Is 1 or 2 for manifold meshes
        result += vertex_curvature_along_edge( vert, edge ) * faces
        count += faces
    return result / count

# MINMAX method: get the average between min and max curvature
def vertex_curvature_minmax( vert ):
    curvatures = [vertex_curvature_along_edge( vert, edge ) for edge in vert.link_edges]
    return (min(curvatures) + max(curvatures)) / 2
    
# Calculate the curvature for all vertices
# Return the curvatures and a table of vertex indices which correspond 
# to a curvature above the threshold 
def mean_curvature( obj, bm, method = 'AVG'):
    if method == 'AVG':
        method_func = vertex_curvature_avg
    elif method == 'MINMAX':
        method_func = vertex_curvature_minmax
    else:
        raise Exception( 'Bad method: should be \'AVG\' or \'MINMAX\'' )

    # Get bmesh access on the mesh
    #bm = bmesh.new()
    #bm.from_mesh( obj.data )

    #bm.verts.ensure_lookup_table()
    #bm.edges.ensure_lookup_table()

    #above_threshold = {}
    curvatures = []
    for vert in bm.verts:
        curvature = method_func( vert )
        curvatures.append( curvature )
        #if abs(curvature) > threshold:
            #above_threshold[vert.index] = True
    return curvatures
    #return curvatures, above_threshold
    
def get_concavity_aware_laplacian_matrix(obj,bm):
    curvatures = mean_curvature( obj, bm, method = 'MINMAX')
    is_concave = concave_list(obj ,5)
    gama = 0.0001
    beta = 0.000001
    row_indices = []
    col_indices = []
    weight = []
    for vert in bm.verts:
        pv = vert.co
        vid = vert.index
        for edge in vert.link_edges:
            vvert = edge.other_vert(vert)    # Return the other vertex on this edge
            ppv = vvert.co
            vvid = vvert.index
            elen = (ppv - pv).length
            w = elen / (abs(curvatures[vid] + curvatures[vvid]) + gama)
            if is_concave[vid] or is_concave[vvid]:
                w *= beta
            row_indices.append(vid)
            col_indices.append(vvid)
            weight.append(-w)
    diag = [0] * len(bm.verts)
    for i in range(len(weight)):
        diag[row_indices[i]] -= weight[i]
    for i in range(len(diag)):
        row_indices.append(i)
        col_indices.append(i)
        weight.append(diag[i])
    #weight = np.array(weight)
    return row_indices, col_indices, weight


def get_index_of_selected_vertices():
    mode = bpy.context.active_object.mode
    # Keep track of previous mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Go into object mode to update the selected vertices
    obj = bpy.context.object
    # Get the currently select object
    sel = np.zeros(len(obj.data.vertices), dtype=np.bool)
    # Create a numpy array with empty values for each vertex
    obj.data.vertices.foreach_get('select', sel)
    bpy.ops.object.mode_set(mode=mode)
    return np.where(sel==True)[0]

#def seg_one_teeth():
    #cons = 

def seg_one_tooth(obj, bm):
    cons =[]
    #cnt = 0
    vhs = get_index_of_selected_vertices()
    for i in vhs:
        vert = bm.verts[i]
        for edge in vert.link_edges:
            vvert = edge.other_vert(vert)
            cons.append({vvert.index:1})
    vhset = set()
    for vert in bm.verts:
        if vert.is_boundary:
            cons.append({vert.index:0})
#    for vert in bm.verts:
#        if vert.is_boundary and cnt < 1:
#            cons.append({vert.index:0})
#            cnt += 1
            for edge in vert.link_edges:
                vvert = edge.other_vert(vert)
                if ((vvert.is_boundary == False) and (vvert.index not in vhset)):
                    vhset.add(vvert.index)
                    cons.append({vvert.index:0})
    harmonic_field = compute_concavity_aware_harmonic_field(obj, bm, cons)
    return harmonic_field                 


def compute_concavity_aware_harmonic_field(obj, bm, cons):
    row_indices, col_indices, weight = get_concavity_aware_laplacian_matrix(obj,bm)
    l = len(bm.verts)
    b = np.zeros(l, dtype=np.float64)
    alpha = 10**8
    for i in cons:
        k, = i
        v, = i.values()
        row_indices.append(k)
        col_indices.append(k)
        weight.append(alpha)
        b[k] = v * alpha
    weight = np.array(weight)
    L = scipy.sparse.csr_matrix(
        (weight, (row_indices, col_indices)), shape=(l, l))
    #b = np.array(b)
    harmonic_field = scipy.sparse.linalg.spsolve(L, b)
    return harmonic_field
    

obj = bpy.context.active_object
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
harmonic_field = seg_one_tooth(obj, bm)
#print(harmonic_field)
vertex_group = obj.vertex_groups.new( name = '123' )
for i in range(len(obj.data.vertices)):
    vertex_group.add( [i], 0, 'ADD' )
for i, vert in enumerate(obj.data.vertices):
    vertex_group.add( [vert.index], harmonic_field[i], 'REPLACE' )
#for i, vert in enumerate(obj.data.vertices):
#    if harmonic_field[i] >= 0.5:
#        vert.hide = True

bm.free()


