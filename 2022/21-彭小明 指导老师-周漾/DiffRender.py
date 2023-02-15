import torch
import trimesh
import numpy as np
import imageio

from trimesh.ray import ray_triangle
#from trimesh.ray import ray_pyembree  #使用ray_pyembree加速




debug = False
#render resolution
resy=960
resx=1280
Float = torch.float64
device='cuda'
extIOR, intIOR = 1.00029, 1.5

@torch.jit.script  #torch的脚本函数，支持高速编译如C++等
def dot(v1:torch.Tensor, v2:torch.Tensor, keepdim:bool = False):  #向量做内积
    ''' v1, v2: [n,3]'''
    result = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] + v1[:,2]*v2[:,2]
    if keepdim:
        return result.view(-1,1)   #tensor.view，tensor的维度，-1代表原维度
    return result

@torch.jit.script
def Reflect(wo, n):  #依据入射方向和法线方向，计算反射方向，wo和出射方向都是向外。
    return -wo + 2 * dot(wo, n, True) * n

@torch.jit.script
def Refract(wo:torch.Tensor, n, eta):    #依据出射方向和法线方向还有折射率之比，计算折射光线的方向
    eta = eta.view(-1,1)
    cosThetaI = dot(n, wo, True)
    sin2ThetaI = (1 - cosThetaI * cosThetaI).clamp(min = 0)  #tensor.clamp限制输出范围
    sin2ThetaT = eta * eta * sin2ThetaI
    totalInerR = (sin2ThetaT >= 1).view(-1)  #反射光线
    cosThetaT = torch.sqrt(1 - sin2ThetaI.clamp(max = 1))
    wt = eta * -wo + (eta * cosThetaI - cosThetaT) * n  #计算折射光线方向的公式

    # wt should be already unit length, Numerical error?
    wt = wt / wt.norm(p=2, dim=1, keepdim=True) #一行一个实例，将向量标准化

    return totalInerR, wt

@torch.jit.script
def FrDielectric(cosThetaI:torch.Tensor, etaI, etaT):   #计算菲涅尔项

    sinThetaI = torch.sqrt( (1-cosThetaI*cosThetaI).clamp(0, 1))
    sinThetaT = sinThetaI * etaI / etaT
    totalInerR = sinThetaT >= 1
    cosThetaT = torch.sqrt( (1-sinThetaT*sinThetaT).clamp(min = 0))
    Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT))
    Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT))
    R = (Rparl * Rparl + Rperp * Rperp) / 2
    return totalInerR, R


@torch.jit.script
def JIT_Dintersect(origin:torch.Tensor, ray_dir:torch.Tensor, triangles:torch.Tensor, normals:torch.Tensor):
    '''
        differentiable ray-triangle intersection，计算光线与三角形的可微交点,进来的是光线的原点，方向以及相交三角形的信息；得到的是交点的重心坐标，以及光线上的参数t，这个三角形的法向量
        # <Fast, Minimum Storage Ray/triangle Intersection>
        # https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    '''
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]  #三角形顶点坐标，triangles应该是n*3*3

    # Find vectors for two edges sharing v[0]，三角形的边向量
    edge1 = v1-v0
    edge2 = v2-v0

    pvec = torch.cross(ray_dir, edge2) #两个向量做叉乘
    # If determinant is near zero, ray lies in plane of triangle
    det = dot(edge1, pvec)
    inv_det = 1/det
    # # Calculate distance from v[0] to ray origin
    tvec = origin - v0
    # Calculate U parameter
    u = dot(tvec, pvec) * inv_det
    qvec = torch.cross(tvec, edge1)
    # Calculate V parameter
    v = dot(ray_dir, qvec) * inv_det   #u,v代表交点在三角形中的重心坐标(三角形中某点的坐标可以通过重心坐标插值得到)，t代表从光源出发到交点的距离。
    # Calculate T
    t = dot(edge2, qvec) * inv_det


    n = torch.cross(edge1, edge2)
    n = n / n.norm(dim=1, p=2, keepdim=True)  #一行一个实例
    return u, v, t, n

@torch.jit.script
def JIT_area(triangles):    #算三角形的法向量
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]  
    edge1 = v1-v0
    edge2 = v2-v0
    area = torch.cross(edge1,edge2).norm(p=2, dim=1) #L2范数
    return area

@torch.jit.script
def JIT_area_var(triangles):  #算法向量的方差
    area = JIT_area(triangles)
    area_ave = area.mean().detach() #阻止梯度反向传播
    area_var = ((area-area_ave)/area_ave).pow(2).mean()
    return area_var

@torch.jit.script
def JIT_edge_var(vertices, edge): #算三角形边的方差
    e1 = vertices[edge[:,0]]
    e2 = vertices[edge[:,1]]
    edge_len = (e1-e2).norm(p=2,dim=1)
    edge_ave =  edge_len.mean().detach()
    edge_var = ((edge_len-edge_ave)/edge_ave).pow(2).mean()
    return edge_var

@torch.jit.script
def edge_face_norm(vertices, E2F):  #共用同一条边的两个相邻三角形的法向量
    faces = E2F #[Ex2x3]
    v0 = vertices[faces[:,0,0]]  #E2F:n*2*3*3
    v1 = vertices[faces[:,0,1]]
    v2 = vertices[faces[:,0,2]]
    EF1N = torch.cross(v1-v0, v2-v0) #[Ex3]
    EF1N = EF1N / EF1N.norm(p=2, dim=1, keepdim=True)

    v0 = vertices[faces[:,1,0]]
    v1 = vertices[faces[:,1,1]]
    v2 = vertices[faces[:,1,2]]
    EF2N = torch.cross(v1-v0, v2-v0) #[Ex3]    
    EF2N = EF2N / EF2N.norm(p=2, dim=1, keepdim=True)
    return EF1N, EF2N

@torch.jit.script
def JIT_corner_angles(triangles):  #算出三角形的三个内角的数值
    u = triangles[:, 1] - triangles[:, 0]
    v = triangles[:, 2] - triangles[:, 0]
    w = triangles[:, 2] - triangles[:, 1] #三角形三条边的向量

    face_N = torch.cross(u,v)
    face_N = face_N / face_N.norm(dim=1, p=2, keepdim=True)

    u = u / u.norm(dim=1, p=2,keepdim=True)
    v = v / v.norm(dim=1, p=2,keepdim=True)
    w = w / w.norm(dim=1, p=2,keepdim=True)
    face_angles = torch.empty_like(triangles[:,:,0]) #n*3

    # clip to make sure we don't float error past 1.0
    face_angles[:, 0] = torch.acos(torch.clamp(dot(u, v), -1, 1)) #根据余弦值反算出以弧度表示的角度。
    face_angles[:, 1] = torch.acos(torch.clamp(dot(-u, w), -1, 1))
    # the third angle is just the remaining
    face_angles[:, 2] = np.pi - face_angles[:, 0] - face_angles[:, 1]
    corner_angles = face_angles.view(-1)   #变成一列


    return corner_angles, face_N

class primary_edge_sample(torch.autograd.Function):  #算edge损失以及要调整的edge的索引
    @staticmethod
    def forward(ctx, E_pos, intersect_fun, camera_M, ray_origin):
        assert ray_origin.dim() == 1   #assert是一个断言，后接一个bool表达式，为真，继续执行，否则终止程序
        num = len(E_pos)
        R, K, R_inverse, K_inverse = camera_M   #相机矩阵
        # E_pos [nx2x2]
        ax = E_pos[:,0,0]
        ay = E_pos[:,0,1]
        bx = E_pos[:,1,0]
        by = E_pos[:,1,1]

        #  just sample mid point for now
        x = (ax+bx)/2
        y = (ay+by)/2   #图像平面上，三角形一条边的中点位置
        sample_point = torch.stack((x,y), dim=1) #[nx2]

        Nx = ay-by # (ay - by)x
        Ny = bx-ax # (bx - ax)y
        N = torch.stack((Nx,Ny), dim=1) #[nx2]
        normalized_N = N / N.norm(dim=1, keepdim=True)  #图像平面上三角形某条边的单位外法向量
        eps = 1
        fu_point = sample_point + eps*normalized_N #[nx2]
        fl_point = sample_point - eps*normalized_N #[nx2]  #中点往法向量方向，内外移动

        f_point = torch.cat((fu_point,fl_point), dim=0).T #[2x2n],上下堆叠
        W = torch.ones([1, f_point.shape[1]], dtype=Float, device=device)
        camera_p = K_inverse @ torch.cat([f_point, W], dim=0) # pixel at z=1
        camera_p = torch.cat([camera_p, W], dim=0)
        world_p = R_inverse @ camera_p #[4x2n]
        world_p = world_p[:3].T #[2nx3]
        ray_dir = world_p - ray_origin.view(-1,3)
        ray_origin = ray_origin.expand_as(ray_dir)  #tensor.expand_as，拓展维度如同ray_dir一样
        _, hitted = intersect_fun(Ray(ray_origin, ray_dir)) #这里采用的是trimesh_interset，输入光线，得到三角形的index，以及相交的index
        mask = torch.zeros(2*num, device=device)
        mask[hitted] = 1
        f = mask[:num] - mask[num:]

        dax = dbx = -Nx
        day = dby = -Ny
        dx = torch.stack((dax,dbx),dim=1)
        dy = torch.stack((day,dby),dim=1)
        dE_pos = torch.stack((dx,dy),dim=2) #[nx2x2]
        dE_pos = dE_pos * f.view(-1,1,1) #[n] --> [nx1x1]
  
        valid_edge = f.abs() > 1e-5
        index = sample_point[valid_edge].to(torch.long)


        output = 0.5 * torch.ones(len(index), device=device)

        ctx.mark_non_differentiable(index)
        ctx.save_for_backward(dE_pos, valid_edge)


        return index, output


    @staticmethod

    def backward(ctx, grad_index, grad_output):
        dE_pos, valid_edge = ctx.saved_variables
        dE_pos[valid_edge] *= grad_output.view(-1,1,1)
        return dE_pos, None, None, None, None

class Ray:
    def __init__(self, origin, direction, ray_ind = None):
        self.origin = origin
        self.direction = direction
        if ray_ind is None:
            self.ray_ind = torch.nonzero(torch.ones(len(origin))).squeeze()
        else:
            self.ray_ind = ray_ind
        assert(len(self.direction)==len(self.ray_ind))


    def select(self, mask):
        mask = mask.to("cpu")
        return Ray(self.origin[mask],self.direction[mask],self.ray_ind[mask])

    def __len__(self):
        return len(self.ray_ind)

class Intersection:
    def __init__(self, u, v, t, n, ray, faces_ind):
        self.u = u
        self.v = v
        self.t = t
        self.n = n
        self.ray = ray
        self.faces_ind = faces_ind
        assert(len(n)==len(ray))
        
    def __len__(self):
        return len(self.ray)

class Scene:  #把光线与物体求交等部分完成
    def __init__(self, mesh_path, cuda_device = 9):   #这里的mesh_path就是三维扫描文件
        self.update_mesh(mesh_path)

    def update_mesh(self, mesh_path):
        print(mesh_path)
        mesh = trimesh.load(mesh_path, process=False)
        assert mesh.is_watertight
        self.mesh = mesh
        self.vertices = torch.tensor(mesh.vertices, dtype=Float, device=device) #三角形的顶点坐标
        self.faces = torch.tensor(mesh.faces, dtype=torch.long, device=device) #构成一个三角形平面的对应顶点的索引
        self.triangles = self.vertices[self.faces] #[Fx3x3] #每个三角形的三个顶点的坐标


        self.ray_triang=ray_triangle.RayMeshIntersector(self.mesh)
        #self.ray_triang = ray_pyembree.RayMeshIntersector(self.mesh)

        self.init_VN()
        self.init_weightM()
        self.init_edge()  #权重初始化等等

    def init_VN(self):  #计算顶点法向量
        faces = self.faces.detach()
        # triangles = self.triangles.detach()
        triangles = self.triangles
        vertices = self.vertices.detach()
        corner_angles, face_N = JIT_corner_angles(triangles) #算出三角形的三个内角数值，以及法向量
        if torch.isnan(corner_angles).any():
            print("nan in corner_angles")
        if torch.isnan(face_N).any():
            print("nan in face_N")
        row = faces.view(-1)
        col = torch.arange(len(faces), device=device).unsqueeze(1).expand(-1,3).reshape(-1)
        coo = torch.stack((row,col))
        weight = corner_angles.detach()
        ver_angle_M = torch.sparse.FloatTensor(coo, weight, torch.Size([len(vertices), len(faces)]))
        vert_N = ver_angle_M.mm(face_N) 
        self.normals = vert_N / vert_N.norm(dim=1, p=2, keepdim=True)

    def init_edge(self):
        '''
        # Calculate E2V_index for silhouette detection
        '''
        mesh = self.mesh
        e1 = mesh.vertices[mesh.edges[:,0]]
        e2 = mesh.vertices[mesh.edges[:,1]]
        self.mean_len = np.linalg.norm(e1-e2, axis=1).mean()

        Egroups = trimesh.grouping.group_rows(mesh.edges_sorted, 2)
        # unique, undirectional edges
        edges = mesh.edges_sorted[Egroups[:,0]]
        E2F_index = mesh.edges_face[Egroups] #[Ex2]
        E2F = self.faces[E2F_index] #[Ex2x3]
        Edges = torch.tensor(edges, device=device)
        self.Edges = Edges
        self.E2F = E2F

    def init_weightM(self):
        '''
        # Calculate a sparse matrix for laplacian operations
        '''
        neighbors = self.mesh.vertex_neighbors
        col = np.concatenate(neighbors)
        row = np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])
        weight = np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])
        col = torch.tensor(col, device=device)
        row = torch.tensor(row, device=device)
        coo = torch.stack((row,col))
        weight = torch.tensor(weight, dtype=Float, device=device)
        size = len(self.vertices)
        self.weightM = torch.sparse.FloatTensor(coo, weight, torch.Size([size, size]))


    def update_verticex(self, vertices:torch.Tensor):
        #trimesh中ray_triang的网格的顶点坐标更新，但相应的面不变
        tri_vertices=vertices.to(torch.float32).to("cpu").detach().numpy()
        self.ray_triang.mesh.vertices=tri_vertices

        self.mesh.vertices = vertices.detach().cpu().numpy()
        self.vertices = vertices
        self.triangles = vertices[self.faces] #[Fx3x3]
        self.init_VN()

    def trimesh_intersect(self, ray:Ray):
        trimesh_o1 = ray.origin.to(torch.float32).to("cpu").detach().numpy()
        trimesh_d1 = ray.direction.to(torch.float32).to("cpu").detach().numpy()
        index_tri = np.zeros(trimesh_o1.shape[0], dtype=int)
        index_tri = self.ray_triang.intersects_first(trimesh_o1, trimesh_d1)  # 这个函数是得到光线第一次相交的三角形网格
        hitted_pai = np.full(index_tri.shape, True, dtype=bool)
        hitted_pai[index_tri == -1] = False
        faces_index = torch.tensor(index_tri, dtype=torch.long).to(device)
        hitted = torch.tensor(hitted_pai).to(device)
        return faces_index, hitted


    def edge_var(self):
        return JIT_edge_var(self.vertices, self.Edges)

    def area_var(self):
        return JIT_area_var(self.triangles)

    def area_sum(self):
        return JIT_area(self.triangles).sum()

    def laplac_hook(self, grad):
        vertices = self.vertices.detach()
        laplac = vertices - self.weightM.mm(vertices) 
        self.hook_rough = torch.norm(laplac, dim=1).abs().mean().item()
        print(self.hook_rough, torch.norm(grad, dim=1).abs().mean().item())
        return self.hook_w * laplac + grad

    def laplac_normal_hook(self, grad):
        vertices = self.vertices.detach()
        laplac = vertices - self.weightM.mm(vertices) 
        laplac = (laplac * self.hook_normal).sum(dim=1, keepdim=True)
        self.hook_rough = laplac.abs().mean().item()
        laplac[laplac.abs()<0.005]=0
        return self.hook_w * laplac + grad

    def render_transparent(self, origin:torch.Tensor, ray_dir:torch.Tensor):
        out_ori = torch.zeros(ray_dir.shape, dtype=Float, device=device)
        out_dir = torch.zeros(ray_dir.shape, dtype=Float, device=device)
        mask = torch.zeros(ray_dir.shape, dtype=torch.bool, device=device)

        ray_ref2 = self.trace2(Ray(origin, ray_dir))
        _, hitted = self.trimesh_intersect(ray_ref2) #折射两次之后的折射光线与那些三角形网格相交,以及相应的光线的index，相交说明光线出不了透明物体，所以要取反
        valid_ray = ray_ref2.select(torch.logical_not(hitted))

        out_ori[valid_ray.ray_ind] = valid_ray.origin
        out_dir[valid_ray.ray_ind] = valid_ray.direction
        mask[valid_ray.ray_ind] = True  #输出折射两次之后不符合要求的的折射光线的信息
        return out_ori, out_dir, mask

    def render_mask(self, origin:torch.Tensor, ray_dir:torch.Tensor):
        _, hitted = self.trimesh_intersect(Ray(origin, ray_dir))
        image = torch.zeros((ray_dir.shape[0]), dtype=Float, device=device)
        image[hitted] = 1  #得到与三角形网格相交的折射光线的标记
        return image
    
    def dihedral_angle(self):
        EF1N, EF2N = edge_face_norm(self.vertices, self.E2F)
        angle = dot(EF1N, EF2N)
        return angle

    def silhouette_edge(self, origin:torch.Tensor):
        assert origin.dim() == 1
        vertices = self.vertices.detach() #[Vx3]
        faces = self.E2F

        EF1N, EF2N = edge_face_norm(vertices, faces)
        F1v = vertices[faces[:,0,0]]
        F2v = vertices[faces[:,1,0]]
        dot1 = dot(EF1N, origin - F1v)
        dot2 = dot(EF2N, origin - F2v)

        silhouette_edge = torch.logical_xor(dot1>0,dot2>0)
        return self.Edges[silhouette_edge]

    def primary_visibility(self, silhouette_edge, camera_M, origin, detach_depth = False):
        '''
            detach_depth: bool
            detach_depth means we don't want the gradient rwt the depth coordinate
        '''
        R, K, R_inverse, K_inverse = camera_M

        V = self.vertices[silhouette_edge.view(-1)] #[2Nx3]
        W = torch.ones([V.shape[0],1], dtype=Float, device=device)
        v_hemo = torch.cat([V, W], dim=1) #[2Nx4]
        v_camera =  R @ v_hemo.T #[4x2N]
        if detach_depth: 
            v_camera[2:3] = v_camera[2:3].detach()
        v_camera = K @ v_camera[:3] #[3x2N]
        pixel_index = v_camera[:2] / v_camera[2]  #[2x2N]
        E_pos = pixel_index.T.reshape(-1,2,2)
        index, output = primary_edge_sample.apply(E_pos, self.trimesh_intersect, camera_M, origin) #[Nx2]

        #out of view
        mask = (index[:,0] < resx-1) * (index[:,1] < resy-1) * (index[:,0] >= 0) * (index[:,1] >= 0)
        return index[mask], output[mask]

    def project_vert(self, camera_M, V:torch.Tensor):
        R, K, R_inverse, K_inverse = camera_M

        W = torch.ones([V.shape[0],1], dtype=Float, device=device)
        v_hemo = torch.cat([V, W], dim=1) #[Nx4]
        v_camera = R @ v_hemo.T #[3xN]
        v_camera = K @ v_camera[:3]
        pixel_index = v_camera[:2] / v_camera[2]
        pixel_index = pixel_index.to(torch.long).T
        return pixel_index

    def Dintersect(self, ray: Ray):
        # hitted, faces = self.trimesh_intersect(origin, ray_dir)，得到一次相交信息
        faces_ind, hitted = self.trimesh_intersect(ray)  #输入光线，得到相交三角形面的index,没有相交的用-1，以及相应的光线的hitted标志
        faces = self.faces[faces_ind[hitted]]
        triangles = self.vertices[faces]
        normals = self.normals[faces] #把与光线相交的三角形信息得到，面，顶点坐标，法向量
        ray_hitted = ray.select(hitted) #挑选出相交的光线信息

        u, v, t, n = JIT_Dintersect(ray_hitted.origin, ray_hitted.direction, triangles, normals)  #return的结果，检测光线与相交三角形数量一致
        return Intersection(u=u, v=v, t=t, n=n, ray=ray_hitted, faces_ind=faces_ind[hitted]), hitted

    def refract_ray(self, intersect: Intersection):
        t = intersect.t
        n = intersect.n
        ray = intersect.ray

        wo = -ray.direction
        cosThetaI = dot(wo, n)
        assert cosThetaI.max()<=1.00001 and cosThetaI.min()>=-1.00001    #, "wo={},n={}".format(*debug_cos())
        cosThetaI = cosThetaI.clamp(-1, 1)
        entering = cosThetaI > 0

        exc = torch.logical_not(entering)
        etaI, etaT = extIOR*torch.ones_like(t), intIOR*torch.ones_like(t)
        etaI[exc], etaT[exc] = etaT[exc], etaI[exc]
        n[exc] = -n[exc]
        cosThetaI[exc] = -cosThetaI[exc]  
        totalInerR1, R = FrDielectric(cosThetaI, etaI, etaT)
        wr = Reflect(wo, n)
        totalInerR2, wt = Refract(wo, n, etaI/etaT)
        if debug:
            assert (totalInerR1 == totalInerR2).all(), (totalInerR1 != totalInerR2).sum()
        refracted = torch.logical_not(totalInerR1)

        new_origin = ray.origin + t.view(-1,1) * ray.direction
        new_dir = wt
        # new_dir = wr
        # TODO: a better way to determine epsilon(1e-5)
        new_origin += 1e-5 * new_dir
        new_ray = Ray(new_origin, new_dir, ray.ray_ind)  #得到折射光线的信息

        return refracted, new_ray

    def trace2(self, ray: Ray):
        intersect, hitted = self.Dintersect(ray)
        refracted, new_ray = self.refract_ray(intersect) #refracted是折射标记
        new_ray = new_ray.select(refracted)

        intersect2, hitted2 = self.Dintersect(new_ray)
        refracted2, new_ray2 = self.refract_ray(intersect2)
        new_ray2 = new_ray2.select(refracted2) #得到折射两次的光线信息

        return new_ray2




