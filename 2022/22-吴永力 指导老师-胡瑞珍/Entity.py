import taichi as ti
import taichi.math as tm
from Transform import *

@ti.dataclass
class Entity:
    invMass:ti.f32
    fixed:ti.i32         #0:dynamic   #1:fixed
    #curr state
    invInertia:tm.vec3
    transform:Transform
    vec:tm.vec3
    omega:tm.vec3
    #last state
    prevTransform:Transform
    #coefficient
    staticFricCoeff:ti.f32
    dynamicFircCoeff:ti.f32
    restitutionCoeff:ti.f32

    @ti.func
    def GetGeneralizedInvMass(self,normal:tm.vec3,r:tm.vec3) -> ti.f32:
        rn = tm.cross(r,normal)
        iMatrix = self.GetWorldInvInertia()
        wI = rn.transpose() @ iMatrix @ rn

        w = self.invMass+wI

        return w

    @ti.func
    def GetWorldInvInertia(self):
        invInertia = self.invInertia
        iMatrix=tm.mat3(invInertia.x,0.0,0.0,
                        0.0,invInertia.y,0.0,
                        0.0,0.0,invInertia.z)
        rotationMat = Quaterion.SetToRotate(self.transform.rotation)
        rotMat3 = tm.mat3(rotationMat[0,0],rotationMat[0,1],rotationMat[0,2],
                          rotationMat[1,0],rotationMat[1,1],rotationMat[1,2],
                          rotationMat[2,0],rotationMat[2,1],rotationMat[2,2])
        return rotMat3 @ iMatrix @ rotMat3.transpose()

    @ti.func
    def GetWorldInertia(self):
        invInertia = self.invInertia
        iMatrix=tm.mat3(1/invInertia.x,0.0,0.0,
                        0.0,1/invInertia.x,0.0,
                        0.0,0.0,1/invInertia.x)
        rotationMat = Quaterion.SetToRotate(self.transform.rotation)
        rotMat3 = tm.mat3(rotationMat[0,0],rotationMat[0,1],rotationMat[0,2],
                          rotationMat[1,0],rotationMat[1,1],rotationMat[1,2],
                          rotationMat[2,0],rotationMat[2,1],rotationMat[2,2])
        return rotMat3 @ iMatrix @ rotMat3.transpose()
        
        
    @ti.func
    def ApplyRotationCorrection(self,corr:tm.vec3):
        if self.fixed == 0:
            iMatrix=self.GetWorldInvInertia()
            iP = iMatrix @ corr
            dq = Quaterion.SetFromValue(iP.x,iP.y,iP.z,0.0)
            dq = 0.5*Quaterion.Multiply(dq,self.transform.rotation)
            #print(f"dq(rotCorr):{dq.x},{dq.y},{dq.z}")
            #atomic add
            self.transform.rotation += dq
            #self.transform.rotation = Quaterion.Normalized(self.transform.rotation)         #doesn't need
    
    @ti.func
    def ApplyPosCorrection(self,corr:tm.vec3,r:tm.vec3):
        if self.fixed == 0:
            #print(f"p:{corr.x},{corr.y},{corr.z}")
            #print(f"r(PosCorr):{r.x},{r.y},{r.z}")
            #atomic add
            corrDelta_x = corr * self.invMass
            self.transform.position += corrDelta_x
            #print(f"deltaX(PosCorr):{corrDelta_x.x},{corrDelta_x.y},{corrDelta_x.z}")
            #self.delta_pos += corrDelta_x
            iMatrix=self.GetWorldInvInertia()
            rp=tm.cross(r,corr)
            #print(f"rp(PosCorr):{rp.x},{rp.y},{rp.z}")
            iRp = iMatrix @ rp
            #print(f"iRp(PosCorr):{iRp.x},{iRp.y},{iRp.z}")
            dq = Quaterion.SetFromValue(iRp.x,iRp.y,iRp.z,0.0)
            dq = 0.5 * Quaterion.Multiply(dq,self.transform.rotation)
            #print(f"dq(PosCorr):{dq.x},{dq.y},{dq.z},{dq.w}")
            #atomic add
            #print(f"rotation(before):{self.transform.rotation.x},{self.transform.rotation.y},{self.transform.rotation.z},{self.transform.rotation.w}")
            self.transform.rotation += dq
            #self.transform.rotation = Quaterion.Normalized(self.transform.rotation)            #doesn't need
            #print(f"rotation(after):{self.transform.rotation.x},{self.transform.rotation.y},{self.transform.rotation.z},{self.transform.rotation.w}")

    @ti.func
    def ApplyVecCorrection(self,corr:tm.vec3,r:tm.vec3):
        self.vec += corr*self.invMass
        rp = tm.cross(r,corr)
        invIMatrix=self.GetWorldInvInertia()

        iDeltaOmega = invIMatrix @ rp

        self.omega += iDeltaOmega
    