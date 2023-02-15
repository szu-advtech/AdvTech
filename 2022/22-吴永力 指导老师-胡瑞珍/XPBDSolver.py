import taichi as ti
import taichi.math as tm
import math
import time
from Entity import *
from Constraints import *
from model import *

gravity=tm.vec3(0.0,-9.8,0.0)

@ti.data_oriented
class XPBDSolver:
    def __init__(self,timestep:ti.f32,numSubStep:ti.f32,entities:list=[]) -> None:
        self.timeStep = timestep
        self.numSubSteps = numSubStep
        self.dt = 2.5e-4
        self.solverEntity = 128
        self.solverConstraint = (self.solverEntity + 1) **2
        self.root = ti.root.pointer(ti.i,1)

        self.entityField = Entity.field(layout=ti.Layout.AOS)
        self.entitySnode = self.root.dynamic(ti.i,self.solverEntity)                   
        self.entitySnode.place(self.entityField)

        self.worldField = tm.mat4.field()
        self.worldSnode = self.root.dynamic(ti.i,self.solverEntity)    
        self.worldSnode.place(self.worldField)

        self.constraintField = Constraints.field(layout=ti.Layout.SOA)
        self.constaintSnode = self.root.dynamic(ti.j,self.solverConstraint)      #max constraint:1024
        self.constaintSnode.place(self.constraintField)

        self.forceField = tm.vec3.field()
        self.forceSnode =self.root.dynamic(ti.j,self.solverConstraint)
        self.forceSnode.place(self.forceField)

        for i in range(len(entities)):
            self.AddEntity(entities[i])
    
    @ti.kernel
    def AddEntity(self,entity:ti.template()):
        ti.append(self.entitySnode,0,entity[None])

    @ti.kernel
    def AddStraint(self,constraint:ti.template()):
        ti.append(self.constaintSnode,0,constraint[None])

    #lambda
    @ti.func
    def GetPositionDeltaLambda(self,body1Id:ti.i32,body2Id:ti.i32,r1:tm.vec3,r2:tm.vec3,
                dt:ti.f32,corr:tm.vec3,compliance:ti.f32) -> ti.f32:
        #get generalized inverse mass
        normal = corr.normalized()
        c = corr.norm()

        w1 = self.entityField[body1Id].GetGeneralizedInvMass(normal,r1)
        w2 = self.entityField[body2Id].GetGeneralizedInvMass(normal,r2)

        #small step
        lambdaCompliance = compliance / (dt**2)
        dlambda = (-c) / (w1+w2+lambdaCompliance)

        return dlambda

    @ti.func
    def GetRotationDeltaLambda(self,body1Id:ti.i32,body2Id:ti.i32,dt:ti.f32,
                corr:tm.vec3,compliance:ti.f32) -> ti.f32:
        #get generalized inverse mass
        axis=tm.normalize(corr)
        magnitude=tm.length(corr)

        body1 = self.entityField[body1Id]
        body2 = self.entityField[body2Id]

        iMatrix1 = body1.GetWorldInvInertia()
        w1= axis.transpose() @ iMatrix1 @ axis

        iMatrix2 = body2.GetWorldInvInertia()
        w2 = axis.transpose() @ iMatrix2 @ axis

        #small step
        lambdaCompliance = compliance/(dt**2)
        dlambda=(-magnitude)/(w1+w2+lambdaCompliance)

        return dlambda

    #first core project correction
    @ti.func
    def ApplyPosCorrectionPair(self,body1Id:ti.i32,body2Id:ti.i32,
                    r1:tm.vec3,r2:tm.vec3,corr:tm.vec3,delta_lamba:ti.f32):
        normal = corr.normalized()
        delta_p = delta_lamba * normal                  #p=lambda * n
        self.entityField[body1Id].ApplyPosCorrection(delta_p,r1)
        self.entityField[body2Id].ApplyPosCorrection(-delta_p,r2)

    #second core project correction
    @ti.func
    def ApplyRotCorrectionPair(self,body1Id:ti.i32,body2Id:ti.i32,corr:tm.vec3,delta_lamba:ti.f32):
        #get generalized inverse mass
        magnitude = corr.normalized()
        delta_p =  delta_lamba * magnitude          #p=lambda * n
        self.entityField[body1Id].ApplyRotationCorrection(delta_p)
        self.entityField[body2Id].ApplyRotationCorrection(-delta_p)

    @ti.kernel
    def preSolve(self,dt:ti.f32,gravity:tm.vec3,moment_ext:tm.vec3):
        for i in ti.grouped(self.entityField):
            if self.entityField[i].fixed == 0:
                #prev
                self.entityField[i].prevTransform = self.entityField[i].transform
                #clac vec
                self.entityField[i].vec += dt * gravity* self.entityField[i].invMass
                self.entityField[i].transform.position += dt * self.entityField[i].vec
                #clac omega
                currOmega=self.entityField[i].omega
                currInertia = self.entityField[i].GetWorldInertia()
                currInvInertia = self.entityField[i].GetWorldInvInertia()
                lw = currInertia @ currOmega
                tau = moment_ext - tm.cross(currOmega,lw)
                dw = dt * (currInvInertia @ tau)
                self.entityField[i].omega += dw

                currOmega = self.entityField[i].omega
                qw = Quaterion.SetFromValue(currOmega[0],currOmega[1],currOmega[2],0)
                dq = dt *0.5 *Quaterion.Multiply(qw,self.entityField[i].transform.rotation)
                self.entityField[i].transform.rotation += dq
                self.entityField[i].transform.rotation = Quaterion.Normalized(self.entityField[i].transform.rotation)

    @ti.kernel
    def UpdateAfterSolve(self,dt:ti.f32):
        for i in ti.grouped(self.entityField):
            if self.entityField[i].fixed == 0:
                delta_x = self.entityField[i].transform.position-self.entityField[i].prevTransform.position
                self.entityField[i].vec = delta_x / dt
                qinv=Quaterion.Conjugate(self.entityField[i].prevTransform.rotation)
                dq=Quaterion.Multiply(self.entityField[i].transform.rotation,qinv)
                domega = (2*tm.vec3(dq.xyz)) / dt
                if dq.w>0 :
                    self.entityField[i].omega = domega
                else:
                    self.entityField[i].omega = -domega

   #*********************Constrain Solver**************************************
    @ti.kernel
    def SolveConstraint(self,dt:ti.f32):
        for i in ti.grouped(self.constraintField):
            #pos constraint
            if self.constraintField[i].conType == 0:
                self.SolveDistanceConstraint(self.constraintField[i],dt)
            #collision constraint
            elif self.constraintField[i].conType == 1:
                self.SolveCollision(self.constraintField[i],dt)
            #joint constraints
            elif self.constraintField[i].conType == 2:
                self.SolveJointsConstraint(self.constraintField[i],dt)
                    
    @ti.func
    def SolveJointsConstraint(self,jointsConstraint:ti.template(),dt:ti.f32):
            body1 =  self.entityField[jointsConstraint.body1Id]
            body2 =  self.entityField[jointsConstraint.body2Id]
            q1 = body1.transform.rotation
            q2 = body2.transform.rotation
            #mutual orientations of two bodies
            if jointsConstraint.jointType == 0:
                q2Inv = Quaterion.Conjugate(q2)
                dq = Quaterion.Multiply(q1,q2Inv)
                omega = 2.0 * dq.xyz
                if dq.w < 0.0:
                    omega = - omega

                if omega.norm() > 0: 
                    delta_lambda = self.GetRotationDeltaLambda(jointsConstraint.body1Id,jointsConstraint.body2Id,dt,omega,
                                                jointsConstraint.compliance)
                    jointsConstraint.lambda_total += delta_lambda * omega.normalized()

                    self.ApplyRotCorrectionPair(jointsConstraint.body1Id,jointsConstraint.body2Id,omega,delta_lambda) 

            #hinge joints
            elif jointsConstraint.jointType == 1:
                a0=Quaterion.GetQuatAixs2(q1)
                a1=Quaterion.GetQuatAixs2(q2)
                
                #align(x)
                dQhinge = tm.cross(a0,a1)
                if dQhinge.norm() > 0:
                    delta_lambda = self.GetRotationDeltaLambda(jointsConstraint.body1Id,jointsConstraint.body2Id,dt,dQhinge,
                                                jointsConstraint.compliance)
                    jointsConstraint.lambda_total += delta_lambda *dQhinge.normalized()

                    self.ApplyRotCorrectionPair(jointsConstraint.body1Id,jointsConstraint.body2Id,dQhinge,delta_lambda) 
                
                #add position constraint
                self.SolveDistanceConstraint(jointsConstraint,dt)

                if jointsConstraint.hasSwingLimit == 1:
                    #update the quaterion
                    n  = Quaterion.GetQuatAixs0(q1)
                    b1 = Quaterion.GetQuatAixs2(q1)
                    b2 = Quaterion.GetQuatAixs2(q2)
                    dQlimit=XPBDSolver.limitAngle(n,b1,b2,
                        jointsConstraint.minSwingAngle,jointsConstraint.maxSwingAngle)
                    if dQlimit.norm() > 0:
                        delta_lambda = self.GetRotationDeltaLambda(jointsConstraint.body1Id,jointsConstraint.body2Id,dt,dQlimit,
                                                jointsConstraint.compliance)

                        self.ApplyRotCorrectionPair(jointsConstraint.body1Id,jointsConstraint.body2Id,dQlimit,delta_lambda) 

            #spherical
            elif jointsConstraint.jointType == 2:
                self.SolveDistanceConstraint(jointsConstraint,dt)
                #swing limits
                if jointsConstraint.hasSwingLimit == 1:
                    #update the quaterion
                    n  = Quaterion.GetQuatAixs2(q1)
                    b1 = Quaterion.GetQuatAixs0(q1)
                    b2 = Quaterion.GetQuatAixs0(q2)
                    dQlimit=XPBDSolver.limitAngle(n,b1,b2,
                        jointsConstraint.minSwingAngle,jointsConstraint.maxSwingAngle)
                    if dQlimit.norm() > 0:
                        delta_lambda = self.GetRotationDeltaLambda(jointsConstraint.body1Id,jointsConstraint.body2Id,dt,dQlimit,
                                                jointsConstraint.compliance)

                        self.ApplyRotCorrectionPair(jointsConstraint.body1Id,jointsConstraint.body2Id,dQlimit,delta_lambda)  

                #twist limits
                if jointsConstraint.hasTwistLimit == 1:
                    #update the quaterion
                    q1= body1.transform.rotation
                    q2= body2.transform.rotation

                    a1=Quaterion.GetQuatAixs0(q1)
                    a2=Quaterion.GetQuatAixs0(q2)

                    b1=Quaterion.GetQuatAixs1(q1)
                    b2=Quaterion.GetQuatAixs1(q2)

                    n=tm.normalize(a1+a2)
                    n1= b1- tm.dot(n,b1) * n
                    n1=tm.normalize(n1)
                    n2= b2- tm.dot(n,b2) * n
                    n2=tm.normalize(n2)

                    #handling gimbal lock proble
                    #maxCorr= 2.0 * math.PI if tm.dot(a1,a2) > -0.5 else 1.0 * self.deltaTime

                    dQlimit = XPBDSolver.limitAngle(n,n1,n2,
                      jointsConstraint.minSwingAngle,jointsConstraint.maxSwingAngle)
                    if dQlimit.norm() > 0:
                        delta_lambda = self.GetRotationDeltaLambda(jointsConstraint.body1Id,jointsConstraint.body2Id,dt,dQlimit,
                                                jointsConstraint.compliance)

                        self.ApplyRotCorrectionPair(jointsConstraint.body1Id,jointsConstraint.body2Id,dQlimit,delta_lambda) 
            



    @ti.func
    def SolveDistanceConstraint(self,posConstraint:ti.template(),dt:ti.f32):
        bodyTransform1 = self.entityField[posConstraint.body1Id].transform
        bodyTransform2 = self.entityField[posConstraint.body2Id].transform
        r1 = Quaterion.Rotate(bodyTransform1.rotation,posConstraint.r1)   
        r2 = Quaterion.Rotate(bodyTransform2.rotation,posConstraint.r2)   
        #print(f"r1:{r1.x},{r1.y},{r1.z}")
        #print(f"r2:{r2.x},{r2.y},{r2.z}")
        #solve distance constraint
        worldPos1 = bodyTransform1.position + r1
        worldPos2 = bodyTransform2.position + r2  
        #print(f"worldPos1:{worldPos1.x},{worldPos1.y},{worldPos1.z}")
        #print(f"worldPos2:{worldPos2.x},{worldPos2.y},{worldPos2.z}")
        ## delta_r = p2-p1
        attachment_distance = worldPos1 - worldPos2
        delta_distance = attachment_distance.norm() - posConstraint.maxDistance
        delta_x = attachment_distance.normalized() * delta_distance
        delta_lambda = self.GetPositionDeltaLambda(posConstraint.body1Id,posConstraint.body2Id,r1,r2,dt,delta_x,
                                    posConstraint.compliance)
        if delta_distance > 0:
            #print(f"delta_distance:{delta_distance}")
            self.ApplyPosCorrectionPair(posConstraint.body1Id,posConstraint.body2Id,r1,r2,delta_x,delta_lambda)
            posConstraint.lambda_total += delta_lambda * delta_x.normalized()

    @ti.func
    def SolveCollision(self,collisoConstraint:ti.template(),dt:ti.f32):
        body1 = self.entityField[collisoConstraint.body1Id]
        body2 = self.entityField[collisoConstraint.body2Id]

        normal = collisoConstraint.contact_normal
        r1 = Quaterion.Rotate(body1.transform.rotation,collisoConstraint.r1)
        r2 = Quaterion.Rotate(body1.prevTransform.rotation,collisoConstraint.r2)
        world_p1 = body1.transform.position + r1
        world_p2 = body2.transform.position + r2
        world_p1_prev = body1.prevTransform.position + r1
        world_p2_prev = body2.prevTransform.position + r2
        
        distance = tm.dot((world_p1 - world_p2),normal)
        #handle contact
        if distance > 0 :
            delta_x = distance * normal

            dlambda_n = self.GetPositionDeltaLambda(collisoConstraint.body1Id,collisoConstraint.body2Id,
                                                       r1,r2,dt,delta_x,0.0)
            #apply collision constraint
            self.ApplyPosCorrectionPair(collisoConstraint.body1Id,collisoConstraint.body2Id,
                                             r1,r2,delta_x,dlambda_n)
            collisoConstraint.lambda_n += dlambda_n
            
            #handle static friction
            dlambda_t = self.GetPositionDeltaLambda(collisoConstraint.body1Id,collisoConstraint.body2Id,
                                                       r1,r2,dt,delta_x,0.0)

            mu_s = (body1.staticFricCoeff + body2.staticFricCoeff)/2
            
            if dlambda_t > dlambda_n * mu_s :
                delta_p = (world_p1-world_p1_prev)-(world_p2-world_p2_prev)
                delta_p_t = delta_p - tm.dot(delta_p,normal) * normal

                #apply static fric
                self.ApplyPosCorrectionPair(collisoConstraint.body1Id,collisoConstraint.body2Id,
                                             r1,r2,delta_p_t,dlambda_t)
                collisoConstraint.lambda_t += dlambda_t

    # #*********************Velocity solver**************************************
    #todo: add joints damping
    @ti.func
    def JointsDamping(self,joints:ti.template()):
        pass

    @ti.kernel
    def SolveVelocity(self,dt:ti.f32):
        #handle collision vec correction
        for i in ti.grouped(self.constraintField):
            if self.constraintField.conType == 1:
                body1 = self.entityField[self.constraintField[i].body1Id]
                body2 = self.entityField[self.constraintField[i].body2Id]
                r1 = self.constraintField[i].r1
                r2 = self.constraintField[i].r2
                normal = self.constraintField[i].contact_normal
                lambda_n =self.constraintField[i].lambda_n
                vec = (body1.vec + tm.cross(body1.omega,r1)) - (body2.vec + tm.cross(body2.omega,r2))
                vec_n = tm.dot(normal,vec)
                vec_t = vec - vec_n * normal

                mu_d=(body1.dynamicFircCoeff + body2.dynamicFircCoeff) / 2
                #f_n = lambda_n / dt^2
                f_n = lambda_n / (dt**2)
                deltaVec = -tm.normalize(vec_t) * ti.min(dt * mu_d * f_n,tm.length(vec_t)) 

                e=(body1.restitutionCoeff + body2.restitutionCoeff) / 2
                if vec_n < 2 * tm.length(gravity) * dt:
                     deltaVec += normal * (-vec_n)
                else:
                    deltaVec += normal * ( -vec_n + tm.min(e*vec_n,0.0))

                normal=tm.normalize(deltaVec)
                #apply deltaVec
                w1 = body1.GetGeneralizedInvMass(normal,r1)
                w2 = body2.GetGeneralizedInvMass(normal,r2)
                #p = delta_v / (w1+w2)
                corr = deltaVec / (w1+w2)
                if body1.fixed == 0:
                    body1.ApplyVecCorrection(corr,r1)
                if body2.fixed == 0:
                    body1.ApplyVecCorrection(corr,r2)

    #### joints limit
    @staticmethod
    @ti.func
    def limitAngle(n:tm.vec3,n1:tm.vec3,n2:tm.vec3,alpha:ti.f32,beta:ti.f32) -> tm.vec3:
        rhi=tm.asin(tm.dot(tm.cross(n1,n2),n))
        if tm.dot(n1,n2) < 0:
            rhi=math.pi-rhi     
        if rhi >math.pi:
            rhi=rhi-2*math.pi
        if rhi < -math.pi:
            rhi=rhi+2*math.pi
        
        dQlimit = tm.vec3(0,0,0)
        if rhi < alpha or rhi > beta :
            rhi=tm.clamp(rhi,alpha,beta)
            #rot(n,rhi)
            q = Quaterion.SetFromAxisAngle(n,rhi)
            #n1 <- rot(n,rhi)n1
            n1 = Quaterion.Rotate(q,n1)

            dQlimit += tm.cross(n1,n2)

        return dQlimit

    #set iterter nnum
    def SetDtAndNumsubStep(self,timestep,numsubsteps):
        self.timeStep=timestep
        self.numSubSteps=numsubsteps
        self.dt = self.timeStep / self.numSubSteps

    @ti.kernel
    def InitConstraintLambda(self):
        for i in ti.grouped(self.constraintField):
            self.constraintField[i].lambda_total=tm.vec3(0.0,0.0,0.0)


    #iterator solver
    def Solver(self):
        totaltime=0
        if self.dt  > 0 :
            #self.InitConstraintLambda()
            #todo:collision detect(board-phase) 
            for iter in range(self.numSubSteps):
                self.InitConstraintLambda()

                start_time = time.time()
                #prev pos
                self.preSolve(self.dt ,gravity,tm.vec3(0.0,0.0,0.0))
                #constraint rigid body
                self.SolveConstraint(self.dt)
                #get vec
                self.UpdateAfterSolve(self.dt)
                #solve vecolity
                self.SolveVelocity(self.dt)
                end_time = time.time()
                totaltime +=(end_time - start_time)
        
        #print(f"solver time:{totaltime*1000}ms")

    #return world matrix
    @ti.kernel
    def UpdataWorldMatrix(self):
        for i in ti.grouped(self.entityField):
            self.worldField[i] = self.entityField[i].transform.GetWorldMatrix()

    def GetSolverData(self):
        self.UpdataWorldMatrix()
        return self.worldField

    @ti.kernel
    def UpdateForceFiled(self,dt:ti.f32):
        for i in ti.grouped(self.constraintField):
            self.forceField[i] = self.constraintField[i].lambda_total / (dt**2)
        
    def GetForceFiled(self):
        self.UpdateForceFiled(self.dt)
        return self.forceField
