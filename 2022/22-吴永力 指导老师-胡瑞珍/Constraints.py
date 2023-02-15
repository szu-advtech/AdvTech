import taichi as ti
import taichi.math as tm

@ti.dataclass
class Constraints:
    conType:ti.i32           #0: pos 1:coll  2:joint

    body1Id:ti.i32
    body2Id:ti.i32
    r1:tm.vec3
    r2:tm.vec3

    compliance:ti.f32     #for evert constraint
    #pos constraint
    maxDistance:ti.f32    # for position constraint
    lambda_total:tm.vec3   # for calc force

    #collision constraint
    contact_normal:tm.vec3
    lambda_n:ti.f32
    lambda_t:ti.f32

    #Joints  constraint
    jointType:ti.f32          #0: fixed  1:hinge 2:sphere
    jointCompliance:ti.f32
    #hinge
    axes:ti.i32             #0:x  1:y  22:z
    #Swing Limit
    hasSwingLimit:ti.i32
    minSwingAngle:ti.f32
    maxSwingAngle:ti.f32
    SwingLimitCompliance:ti.f32    
    #TwingLimit
    hasTwistLimit:ti.i32
    minTwistAngle:ti.f32
    maxTwistAngle:ti.f32
    TwistLimitCompliance:ti.f32

    def InitDistanceConstraint(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,
        maxDistance:tm.vec3,compliance:ti.f32):
        self.conType = 0
        self.body1Id = body1Id
        self.body2Id = body2Id

        self.r1= r1
        self.r2= r2

        self.maxDistance=maxDistance
        self.compliance=compliance

    def InitCollisionConstraint(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,normal:tm.vec3):
        self.conType = 1
        self.body1Id = body1Id
        self.body2Id = body2Id

        self.r1= r1
        self.r2= r2
        self.contact_normal=normal
        self.lambda_t=0
        self.lambda_n=0
        self.compliance=0

    def InitFixedConstraint(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,maxDistance:ti.f32,compliance:ti.f32):
        self.conType = 2
        self.jointType = 0
        self.body1Id = body1Id
        self.body2Id = body2Id

        self.r1= r1
        self.r2= r2
        self.maxDistance = maxDistance
        self.compliance = compliance

    def InitHingeConstraint(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,
                    maxDistance:ti.f32,
                    compliance:ti.f32,
                    axes=0,
                    hasSwingLimit=0,
                    minSwingAngle=0.0,
                    maxSwingAngle=0.0,
                    SwingLimitCompliance=0.0):
        
        self.conType = 2
        self.jointType = 1
        self.body1Id = body1Id
        self.body2Id = body2Id

        self.r1= r1
        self.r2= r2
        self.maxDistance = maxDistance
        self.axes =axes
        self.jointCompliance = compliance
        self.hasSwingLimit=hasSwingLimit
        self.minSwingAngle=minSwingAngle
        self.maxSwingAngle=maxSwingAngle
        self.SwingLimitCompliance = SwingLimitCompliance

    def InitSphericalConstraint(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,
                    hasSwingLimit:ti.i32,
                    minSwingAngle:ti.f32,
                    maxSwingAngle:ti.f32,
                    SwingLimitCompliance:ti.f32,
                    hasTwistLimit:ti.i32,
                    minTwistAngle:ti.f32,
                    maxTwistAngle:ti.f32,
                    TwistLimitCompliance:ti.f32
                    ):
        self.conType = 2
        self.jointType = 2
        self.body1Id = body1Id
        self.body2Id = body2Id

        self.r1= r1
        self.r2= r2
        self.hasSwingLimit=hasSwingLimit
        self.minSwingAngle=minSwingAngle
        self.maxSwingAngle=maxSwingAngle
        self.SwingLimitCompliance = SwingLimitCompliance

        self.hasTwistLimit = hasTwistLimit
        self.minTwistAngle = minTwistAngle
        self.maxTwistAngle = maxTwistAngle
        self.TwistLimitCompliance = TwistLimitCompliance


        





