import taichi as ti
import taichi.math as tm
from Entity import *
from vispy import io

@ti.data_oriented
class Model:
    def __init__(self,objfile,max_vertex_num=1024, max_faces_num=1024, color=ti.Vector([0.0, 0.0, 0.0])):
        self.vertex_num = ti.field(dtype=ti.i32, shape=())
        self.vertex_num[None] = 0
        self.face_num = ti.field(dtype=ti.i32, shape=())
        self.face_num[None] = 0
        self.indices_num = ti.field(dtype=ti.i32, shape=())
        self.indices_num[None] = 0
        self.vertex = None
        self.face = None
        self.indices = None
        self.normal = None
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.color[None] = color
        self.from_obj(objfile)

    def set_color(self, color):
        self.color[None] = color

    def set_material(self, material):
        self.material[None] = material

    def add_face(self, vertex):
        for index in range(3):
            self.vertex[self.vertex_num[None] + index] = ti.Vector(
                [vertex[index, 0], vertex[index, 1], vertex[index, 2], 1])
        self.face[self.face_num[None]] = ti.Vector(
            [self.vertex_num[None], self.vertex_num[None] + 1, self.vertex_num[None] + 2])
        self.vertex_num[None] += 3
        self.face_num[None] += 1

    def clear_face(self):
        self.vertex_num[None] = 0
        self.face_num[None] = 0

    def from_obj(self, filename):
        self.clear_face()
        vertex, face, normal, texture = io.read_mesh(filename)
        self.vertex_num[None] = len(vertex)
        self.face_num[None] = len(face)
        self.indices_num[None] = len(face) * 3
        self.vertex = ti.Vector.field(4, dtype=ti.f32, shape=len(vertex))
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertex))
        self.face = ti.Vector.field(3, dtype=ti.i32, shape=len(face))
        self.indices =ti.field(dtype=ti.i32,shape=len(face)*3)
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=len(vertex))
        for index in range(len(vertex)):
            self.vertex[index] = ti.Vector([vertex[index][0], vertex[index][1], vertex[index][2], 1])
            self.vertices[index] = ti.Vector([vertex[index][0], vertex[index][1], vertex[index][2]])
            self.normal[index] = ti.Vector([normal[index][0], normal[index][1], normal[index][2]])
        for index in range(len(face)):
            self.face[index] = ti.Vector([face[index][0], face[index][1], face[index][2]])
            self.indices[index * 3] = face[index][0]
            self.indices[index * 3 + 1] = face[index][1]
            self.indices[index * 3 + 2] = face[index][2]

    
    @ti.kernel
    def GetVertexFormWolrdMat(self,worldMatrix:tm.mat4,vertices:ti.template()):
        for index in range(self.vertex_num[None]):
            worldVertex = worldMatrix @ self.vertex[index]
            vertices[index] = tm.vec3(worldVertex.xyz)


    @ti.kernel 
    def GetIndices(self,indices:ti.template()):
        for index in range(self.indices_num[None]):
            indices[index] = self.indices[index]

    #get inertia(diag)
    @ti.kernel
    def GetInertia(self,rho:ti.f32)->tm.mat3:
        #set center_x = (0,0,0)
        #select point
        w0 = self.vertex[0]
        cMatrix_total=ti.Matrix.zero(dt=ti.f32,n=3,m=3)
        cMatrix_nor = tm.mat3([1.0/60.0,1.0/120.0,1.0/120.0],
                            [1.0/120.0,1.0/60.0,1.0/120.0],
                            [1.0/120.0,1.0/120.0,1.0/60.0])
        center_x = tm.vec3(0,0,0)
        mass_total=0.0
        for index in range(self.face_num[None]):
            w10 =  self.vertex[self.face[index][0]] - w0
            w20 =  self.vertex[self.face[index][1]] - w0
            w30 =  self.vertex[self.face[index][2]] - w0
            aMatirx = tm.mat3(w10.xyz,w20.xyz,w30.xyz)
            aDet = aMatirx.determinant()
            cMat_t = aDet * aMatirx @ cMatrix_nor @ aMatirx.transpose()
            #center x
            center_x_index = ((self.vertex[self.face[index][0]]+self.vertex[self.face[index][1]]+self.vertex[self.face[index][2]]+w0)/4).xyz
            mass = rho * (1.0 / 6.0) * aDet 
            delta_x = w0.xyz
            cMat_t = cMat_t +  mass * (delta_x @ center_x_index.transpose() + center_x_index @ delta_x.transpose() + delta_x @ delta_x.transpose())
            cMatrix_total += cMat_t

            center_x += center_x_index * mass
            mass_total += mass

        center_x /= mass_total
        print(center_x)
        print(mass_total)
        trace=cMatrix_total.trace() 
        inertia = ti.Matrix.diag(dim=3,val=trace) - cMatrix_total
        print(inertia)
        return inertia

@ti.data_oriented
class RigidBody:
    def __init__(self,position:tm.vec3,mass:ti.f32,model=None,fixed=0,quaterion=Quaterion.Identity(),
                    scale=tm.vec3(1,1,1),
                    inertia=tm.vec3(1,1,1),vec=tm.vec3(0,0,0),
                    omega=tm.vec3(0,0,0),staticFric=0,dynmanicFric=0,restitutionC=0) -> None:
        self.model = model
        if model is not None:
            self.worldVectices =  ti.Vector.field(3, dtype=ti.f32, shape= self.model.vertex_num[None])
        else:
            self.worldVectices = ti.Vector.field(3, dtype=ti.f32, shape= 1)
            self.worldVectices[0] = position
        self.entity = RigidBody.CreateEntity(position,mass,fixed,inertia,quaterion,scale,
                        vec,omega,staticFric,dynmanicFric,restitutionC)

    @ti.kernel
    def SetRotationLocationZ(self,radians:ti.f32) :
        axis = self.entity[None].transform.GetWorldMatrix() @ tm.vec4(0,0,1,0)
        q = Quaterion.SetFromAxisAngle(axis.xyz,radians)
        self.entity[None].transform.rotation = Quaterion.Multiply( self.entity[None].transform.rotation,q)


    @ti.kernel
    def SetRotationLocationX(self,radians:ti.f32) :
        axis = self.entity[None].transform.GetWorldMatrix() @ tm.vec4(1,0,0,0)
        q = Quaterion.SetFromAxisAngle(axis.xyz,radians)
        self.entity[None].transform.rotation = Quaterion.Multiply( self.entity[None].transform.rotation,q)

    @ti.kernel
    def GetWorlMatrix(self)->tm.mat4:
        return self.entity[None].transform.GetWorldMatrix()

    def GetModelData(self,matrix=None):
        if self.model is None:
            vectex=tm.vec4(self.worldVectices[0],1.0)
            self.worldVectices[0] = tm.vec3((matrix @ vectex).xyz)
            return self.worldVectices,None,None,None
        else:
            self.model.GetVertexFormWolrdMat(matrix,self.worldVectices)
        color = (self.model.color[None].x,self.model.color[None].y,self.model.color[None].z)
        return self.worldVectices,self.model.indices,self.model.normal,color

    @staticmethod
    def CreateEntity(position:tm.vec3,mass:ti.f32,fixed=0,inertia=tm.vec3(1,1,1),quaterion=Quaterion.Identity(),
                    scale=tm.vec3(1,1,1),
                vec=tm.vec3(0,0,0),omega=tm.vec3(0,0,0),staticFric=0,dynmanicFric=0,restitutionC=0):
        data = Entity.field(shape=())
        data[None].fixed = fixed
        if mass == 0:
            data[None].invMass = 0.0
        else:
            data[None].invMass = 1.0 / mass
        if inertia[1] == 0:
            data[None].invInertia = tm.vec3(1/inertia[0],inertia[1],1/inertia[2])
        else:
            data[None].invInertia = tm.vec3(1/inertia[0],1/inertia[1],1/inertia[2])
        wolrdTranform = Transform(position,quaterion,scale)
        data[None].transform = wolrdTranform
        data[None].vec =vec
        data[None].omega= omega
        data[None].staticFricCoeff = staticFric
        data[None].dynamicFircCoeff = dynmanicFric
        data[None].restitutionCoeff = restitutionC
        return data

    @staticmethod
    def GetBoxInertia(mass,depth,width,height)->tm.vec3:
        return tm.vec3((1.0/12.0)*mass*(height**2+depth**2),
                        (1.0/12.0)*mass*(width**2+height**2),
                        (1.0/12.0)*mass*(width**2+depth**2))    
    
    @staticmethod
    def GetSphereInertia(mass,radius)->tm.vec3:
        return tm.vec3((2.0/5.0)*mass*(radius**2),
                        (2.0/5.0)*mass*(radius**2),
                        (2.0/5.0)*mass*(radius**2)) 


    @staticmethod
    def GetRodInertia(mass,length)->tm.vec3:
        return tm.vec3((1.0/3.0)*mass*(length**2),
                        0,
                        (1.0/3.0)*mass*(length**2)) 
