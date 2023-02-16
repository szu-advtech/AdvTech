## transform
import taichi as ti
import taichi.math as tm
import Quaterion 

@ti.dataclass
class Transform:
    #world
    position:tm.vec3
    rotation:tm.vec4
    scale:tm.vec3

    @ti.func
    def GetWorldMatrix(self):
        translateMat=tm.translate(self.position[0],self.position[1],self.position[2])
        rotationMat=Quaterion.SetToRotate(self.rotation)
        scaleMat = tm.scale(self.scale[0],self.scale[1],self.scale[2])
        return translateMat @ rotationMat @ scaleMat