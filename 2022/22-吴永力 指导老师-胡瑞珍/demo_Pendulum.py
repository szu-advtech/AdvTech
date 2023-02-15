import taichi as ti
import time as time
from XPBDSolver import *

ti.init(arch=ti.gpu,offline_cache=True)
RES=(1280,720)
timeStep = 1.0 / 60 
numSubSteps = 5

#
center_x = tm.vec3(0,0,0)
numPendulum = 1
initTopPos = tm.vec3(0,4,0)
#solver rigid body data
bodyList=[]
entities=[]
constraintList=[]

#init scene data
def InitPendulumScene(numPendulum,initPos=tm.vec3(0,0,0),delta_x = 0.6):
    #init floor
    floorModel = Model("models/floor.obj",color=(0.5,0.5,0.5))
    floor = RigidBody(tm.vec3(0,-1,0),1.0,floorModel,fixed=1,quaterion=Quaterion.Identity())
    bodyList.append(floor)
    entities.append(floor.entity)
    lever = Model("models/lever2.obj",color=(0.5,0.5,0.5))
    #constarint init 
    compliance = 0  
    dis = 0.1

    #add shphere
    spherePos = initPos
    sphere = RigidBody(spherePos,1.0,fixed=1)
    bodyList.append(sphere)
    entities.append(sphere.entity)
    r1 = tm.vec3(0,0.0,0)
    r2 = tm.vec3(0,0.0,0)


    for i in range(numPendulum):
        body1Id = len(bodyList) - 1
        if i != 0:
            spherePos += tm.vec3(0.0,delta_x + dis ,0.0)
            r1 = tm.vec3(0,-0.3,0)
        else:
            spherePos += tm.vec3(-0.05,0.0,0.0)
            r1 = tm.vec3(0.0,0.0,0)
        
        body = RigidBody(spherePos,1.0,lever,
                        inertia=RigidBody.GetBoxInertia(1.0,0.6,0.1,0.1))
        #body.SetRotationLocationZ(math.pi/2)
        #body.SetRotationLocationX(math.pi/4)
        bodyList.append(body)
        entities.append(body.entity)
        body2Id = len(bodyList) - 1

        #joint constraint
        constraint = Constraints.field(shape=())
        constraint[None].InitHingeConstraint(body1Id,body2Id,r1,r2,0.05,compliance)
        constraintList.append(constraint)

#init xpbdSolver
def InitXPBDSolver():
    xpbd = XPBDSolver(timeStep,numSubSteps,entities)
    for i in range(len(constraintList)):
        xpbd.AddStraint(constraintList[i])
    return xpbd
        

if __name__ == '__main__':
    window = ti.ui.Window("XPBD with rigid body simulation", RES)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    #init camera
    camera = ti.ui.Camera()
    camera.z_far(1000.0)
    camera.z_near(1.0)
    camera.position(0.0, 4.0, -4.0)
    camera.lookat(0.0,4.0,0.0)
    #init scene mesh
    #InitPendulumScene(2)
    masses = [1.0,10.0]
    InitPendulumScene(numPendulum,initTopPos)
    #init XPBDSolver
    xpbd = InitXPBDSolver()
    
    frame = 0
    isSolving = False
    while window.running:
        gui = window.get_gui()
        with gui.sub_window("gui", 0, 0, 0.3, 0.2):
             numSubSteps = gui.slider_int("numSubSteps",numSubSteps,1,50)
             #isSolving = gui.checkbox("Run or Stop",isSolving)
             isSolving = gui.button("Run or Stop")

        xpbd.SetDtAndNumsubStep(timeStep,numSubSteps)
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.7,0.7,0.7))
        scene.point_light(pos=(0, 10, 0), color=(1.0,1.0,1.0))

        start_time = time.time()
        ################################## logic tick #######################################
        #logic tick
        if isSolving == True:
            xpbd.Solver()
        worldMatData=xpbd.GetSolverData()
        ###################################  render tick  ####################################
        #add mesh to sence(entities)
        for i in range(len(bodyList)):
            worldM = worldMatData[i]
            # draw particles
            if bodyList[i].model == None:
                scene.particles(bodyList[i].worldVectices,0.05)
            else:    
                vertices,indices,normals,color = bodyList[i].GetModelData(worldM)
                scene.mesh(vertices,indices,normals=normals,color=color)
        

        # end_time = time.time()
        # print(f"frame: {frame}, time={(end_time - start_time)*1000:03f}ms")
        canvas.scene(scene)
        window.show()
        frame +=1

        #if frame==2:
            #window.running = False



#ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'
