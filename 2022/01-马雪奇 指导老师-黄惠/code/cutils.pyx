#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Cython specific imports
import numpy as np
cimport numpy as np
import cython
np.import_array()


def dual_contouring_ndc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid):

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles

    # 用于存放顶点和面片的数组是会随着实际顶点数和面片数而动态地增加
    
    # 初始化变量
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384

    # 初始化用于存放顶点和面片的数组
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)

    # 变量定义，将 array 转换为 cdef 形式，好奇这里是值拷贝还是其他？
    cdef float[:,::1]   all_vertices        = all_vertices_
    cdef int[:,::1]     all_triangles       = all_triangles_
    cdef float[:,::1]   all_vertices_old    = all_vertices_
    cdef int[:,::1]     all_triangles_old   = all_triangles_

    cdef int dimx,dimy,dimz
    # 这里的 -1 操作暂时还不太清楚，可能与构建栅格有关系
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    #array for fast indexing vertices 定义一个用于快速索引 vertices 的数组
    vertices_grid_ = np.full([dimx, dimy, dimz], -1, np.int32)
    cdef int[:,:,::1] vertices_grid = vertices_grid_

    cdef int i,j,k, ii, v0,v1,v2,v3,v4,v5,v6,v7


    #all vertices
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i+1,j,k,0]
                v2 = int_grid[i+1,j+1,k,0]
                v3 = int_grid[i,j+1,k,0]
                v4 = int_grid[i,j,k+1,0]
                v5 = int_grid[i+1,j,k+1,0]
                v6 = int_grid[i+1,j+1,k+1,0]
                v7 = int_grid[i,j+1,k+1,0]

                if v1!=v0 or v2!=v0 or v3!=v0 or v4!=v0 or v5!=v0 or v6!=v0 or v7!=v0:
                    #add a vertex
                    vertices_grid[i,j,k] = all_vertices_len

                    #grow all_vertices
                    if all_vertices_len+1>=all_vertices_max:
                        all_vertices_max = all_vertices_max*2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_
                        for ii in range(all_vertices_len):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices
                    all_vertices[all_vertices_len,0] = float_grid[i,j,k,0]+i
                    all_vertices[all_vertices_len,1] = float_grid[i,j,k,1]+j
                    all_vertices[all_vertices_len,2] = float_grid[i,j,k,2]+k
                    all_vertices_len += 1


    #all triangles

    #i-direction
    for i in range(0,dimx):
        for j in range(1,dimy):
            for k in range(1,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i+1,j,k,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1

    #j-direction
    for i in range(1,dimx):
        for j in range(0,dimy):
            for k in range(1,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i,j+1,k,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    #k-direction
    for i in range(1,dimx):
        for j in range(1,dimy):
            for k in range(0,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i,j,k+1,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]




def dual_contouring_undc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid):

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles

    # 用于存放顶点和面片的数组是会随着实际顶点数和面片数而动态地增加
    
    # 初始化变量
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384
    
    # 初始化用于存放顶点和面片的数组
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    
    # 初始化用于存放顶点和面片的数组
    cdef float[:,::1] all_vertices = all_vertices_ # num_v x 3
    cdef int[:,::1] all_triangles = all_triangles_ # num_f x 3
    cdef float[:,::1] all_vertices_old = all_vertices_
    cdef int[:,::1] all_triangles_old = all_triangles_ # 这里我猜测应该是浅拷贝，即 all_triangles 发生变化之后，all_triangles_old 也会发生变化

    cdef int dimx,dimy,dimz
    # 这里的 -1 操作暂时还不太清楚，可能与构建栅格有关系
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    # array for fast indexing vertices
    # 定义一个用于快速索引 vertices 的数组
    vertices_grid_ = np.full([dimx,dimy,dimz], -1, np.int32)
    cdef int[:,:,::1] vertices_grid = vertices_grid_

    cdef int i,j,k, ii
    cdef int i_flag = 1
    #all vertices
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                # v0 的三条边
                v0_0 = int_grid[i,j,k,0]
                v0_1 = int_grid[i,j,k,1] 
                v0_2 = int_grid[i,j,k,2] 

                # v1 的两条边
                v1_1 = int_grid[i+1,j,k,1] 
                v1_2 = int_grid[i+1,j,k,2] 
                
                # v2 的一条边
                v2_2 = int_grid[i+1,j+1,k,2] 

                # v3 的两条边
                v3_0 = int_grid[i,j+1,k,0] 
                v3_2 = int_grid[i,j+1,k,2]

                # v4 的两条边
                v4_0 = int_grid[i,j,k+1,0] 
                v4_1 = int_grid[i,j,k+1,1] 
                
                # v5 的一条边
                v5_1 = int_grid[i+1,j,k+1,1] 

                # v7 的一条边
                v7_0 = int_grid[i,j+1,k+1,0] 

                # 除了 v6 ，其他点的边都多多少少涉及到

                if v0_0 or v0_1 or v0_2 or v1_1 or v1_2 or v2_2 or v3_0 or v3_2 or v4_0 or v4_1 or v5_1 or v7_0: 
                    #add a vertex，这里将 i,j,k 位置的 vertex 的索引储存在数组 vertices_grid[i,j,k]
                    vertices_grid[i,j,k] = all_vertices_len

                    #grow all_vertices
                    if all_vertices_len + 1 >= all_vertices_max:
                        all_vertices_max = all_vertices_max * 2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_

                        # 接下来，先把 old_vertices 拷贝进去
                        for ii in range(all_vertices_len):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        
                        # 然后更新一下 old_vertices
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices, float_grid[i,j,k] 存储着 [i,j,k] "对应的 voxel" 里面的 "vertex 相对于 local origin 的偏差[dx, dy, dz]"
                    if i_flag:
                        all_vertices[all_vertices_len,0] = i + float_grid[i,j,k,0]
                        all_vertices[all_vertices_len,1] = j + float_grid[i,j,k,1]
                        all_vertices[all_vertices_len,2] = k + float_grid[i,j,k,2]
                    else:
                        # 否则调换一下 k 和 j 的顺序
                        all_vertices[all_vertices_len,0] = i + float_grid[i,j,k,0]
                        all_vertices[all_vertices_len,1] = k + float_grid[i,j,k,2]
                        all_vertices[all_vertices_len,2] = j + float_grid[i,j,k,1]

                    all_vertices_len += 1


    #all triangles

    if 1:
        #i-direction，注意，这里是从 0,1,1 开始遍历的
        for i in range(0,dimx):
            for j in range(1,dimy):
                for k in range(1,dimz):
                    # 如果 i,j,k 点的第 0 条边被 surface 切割，那这条边周围的四个 voxel 一定存在连接关系
                    if int_grid[i,j,k,0]:

                        #grow all_triangles
                        if all_triangles_len + 2 >= all_triangles_max:
                            all_triangles_max = all_triangles_max * 2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1

    
    
    
    if 1:
        #j-direction，注意，这里是从 1,0,1 开始遍历的
        for i in range(1,dimx):
            for j in range(0,dimy):
                for k in range(1,dimz):
                    if int_grid[i,j,k,1]:

                        #grow all_triangles
                        if all_triangles_len+2>=all_triangles_max:
                            all_triangles_max = all_triangles_max*2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    if 1:
        #k-direction，注意，这里是从 1,1,0 开始索引的
        for i in range(1,dimx):
            for j in range(1,dimy):
                for k in range(0,dimz):
                    if int_grid[i,j,k,2]:

                        #grow all_triangles
                        if all_triangles_len+2>=all_triangles_max:
                            all_triangles_max = all_triangles_max*2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]

# 这个是用来做测试用的，我们可以只生成某一个方向的所有面片
def dual_contouring_block_undc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid, int[::1] full_scene_size, int idx): 

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles

    # 用于存放顶点和面片的数组是会随着实际顶点数和面片数而动态地增加
    
    # 初始化变量
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384
    
    # 初始化用于存放顶点和面片的数组
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    
    # 初始化用于存放顶点和面片的数组
    cdef float[:,::1] all_vertices = all_vertices_ # num_v x 3
    cdef int[:,::1] all_triangles = all_triangles_ # num_f x 3
    cdef float[:,::1] all_vertices_old = all_vertices_
    cdef int[:,::1] all_triangles_old = all_triangles_ # 这里我猜测应该是浅拷贝，即 all_triangles 发生变化之后，all_triangles_old 也会发生变化

    cdef int dimx,dimy,dimz
    # 这里的 -1 操作暂时还不太清楚，可能与构建栅格有关系
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    # array for fast indexing vertices
    # 定义一个用于快速索引 vertices 的数组
    vertices_grid_ = np.full([dimx,dimy,dimz], -1, np.int32)
    cdef int[:,:,::1] vertices_grid = vertices_grid_

    # 注意，以下参数确实传进来了
    cdef int idx_x   = (idx // (full_scene_size[1] * full_scene_size[2])) * int_grid.shape[0]
    cdef int idx_yz  = idx % (full_scene_size[1] * full_scene_size[2])
    cdef int idx_y   = (idx_yz // full_scene_size[2]) * int_grid.shape[1]
    cdef int idx_z   = (idx_yz % full_scene_size[2]) * int_grid.shape[2]

    cdef int i,j,k, ii

    #all vertices
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                # v0 的三条边
                v0_0 = int_grid[i,j,k,0]
                v0_1 = int_grid[i,j,k,1] 
                v0_2 = int_grid[i,j,k,2] 

                # v1 的两条边
                v1_1 = int_grid[i+1,j,k,1] 
                v1_2 = int_grid[i+1,j,k,2] 
                
                # v2 的一条边
                v2_2 = int_grid[i+1,j+1,k,2] 

                # v3 的两条边
                v3_0 = int_grid[i,j+1,k,0] 
                v3_2 = int_grid[i,j+1,k,2]

                # v4 的两条边
                v4_0 = int_grid[i,j,k+1,0] 
                v4_1 = int_grid[i,j,k+1,1] 
                
                # v5 的一条边
                v5_1 = int_grid[i+1,j,k+1,1] 

                # v7 的一条边
                v7_0 = int_grid[i,j+1,k+1,0] 

                # 除了 v6 ，其他点的边都多多少少涉及到

                if v0_0 or v0_1 or v0_2 or v1_1 or v1_2 or v2_2 or v3_0 or v3_2 or v4_0 or v4_1 or v5_1 or v7_0:
                    #add a vertex，这里将 i,j,k 位置的 vertex 的索引储存在数组 vertices_grid[i,j,k]
                    vertices_grid[i,j,k] = all_vertices_len

                    #grow all_vertices
                    if all_vertices_len + 1 >= all_vertices_max:
                        all_vertices_max = all_vertices_max * 2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_

                        # 接下来，先把 old_vertices 拷贝进去
                        for ii in range(all_vertices_len):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        
                        # 然后更新一下 old_vertices
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices, float_grid[i,j,k] 存储着 [i,j,k] "对应的 voxel" 里面的 "vertex 相对于 local origin 的偏差[dx, dy, dz]"    
                    all_vertices[all_vertices_len,0] = i + idx_x + float_grid[i,j,k,0]
                    all_vertices[all_vertices_len,1] = j + idx_y + float_grid[i,j,k,1]
                    all_vertices[all_vertices_len,2] = k + idx_z + float_grid[i,j,k,2]
                    all_vertices_len += 1


    # all triangles
    # 这里分成三大步进行重建，我的理解是，在每一个triangle生成的时候，是需要考虑其邻近的 voxel 的，如果每个 voxel 同时进行三个方向的重建，可能会比较耗时，不如分成三个方向进行重建
    if 0:
        # i-direction，注意，这里是从 0,1,1 开始遍历的
        for i in range(0,dimx):
            for j in range(1,dimy):
                for k in range(1,dimz):
                    # 如果 i,j,k 点的第 0 条边被 surface 切割，那这条边周围的四个 voxel 一定存在连接关系
                    if int_grid[i,j,k,0]:

                        #grow all_triangles
                        if all_triangles_len + 2 >= all_triangles_max:
                            all_triangles_max = all_triangles_max * 2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1

    if 0:
        #j-direction，注意，这里是从 1,0,1 开始遍历的
        for i in range(1,dimx):
            for j in range(0,dimy):
                for k in range(1,dimz):
                    if int_grid[i,j,k,1]:

                        #grow all_triangles
                        if all_triangles_len+2>=all_triangles_max:
                            all_triangles_max = all_triangles_max*2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    if 0:
        #k-direction，注意，这里是从 1,1,0 开始索引的
        for i in range(1,dimx):
            for j in range(1,dimy):
                for k in range(0,dimz):
                    if int_grid[i,j,k,2]:

                        #grow all_triangles
                        if all_triangles_len+2>=all_triangles_max:
                            all_triangles_max = all_triangles_max*2
                            all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                            all_triangles = all_triangles_
                            for ii in range(all_triangles_len):
                                all_triangles[ii,0] = all_triangles_old[ii,0]
                                all_triangles[ii,1] = all_triangles_old[ii,1]
                                all_triangles[ii,2] = all_triangles_old[ii,2]
                            all_triangles_old = all_triangles_

                        #add to all_triangles
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]