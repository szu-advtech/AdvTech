
from operator import itemgetter
import numpy as np
import json
import math
import turtle
import copy
import cairo
all_nodes=[]
all_links=[]
def readdata(data_path):
    with open (data_path) as f: 
    #利用json.load（）方法将json文件转化为python的dict形式
        layout_data=json.load(f)
    global all_nodes,all_links
    all_nodes=layout_data['nodes']  
    
    all_links=layout_data['links']
    
    return

all_names,all_pos=[],[]
def nodes_out():
#从all_nodes列表中取出取nodes下的pos数据，具体数据是pos下的:x、y坐标
    global all_names,all_pos
    for node in  all_nodes:
        name=node['name']
        pos=node['pos']
        all_names.append(name)
        all_pos.append(pos)
  
source_target_allout=[]
def links_out():
#从all_links取出links中形如{"source": "Napoleon", "target": "Myriel"}的所有对,保存在source_target_allout中
   source_target=['source','target']
   for link in all_links:
      source_target_out={key:link[key] for key in link.keys()if key in source_target}
      source_target_allout.append(source_target_out)
      #(source_target_allout)

source_pos=[]
target_pos=[]
#source_pos_new=[]
#target_pos_new=[]
#soucepos,targetpos分别是从links中取出 source所对应的nodes中的pos、target的pos
def get_sourcetargetpos(HEIGHT_SURFACE):
    source_pos_new=[]
    target_pos_new=[]
#取出source_pos
    global source_pos
    
    for link in all_links:
        for node in all_nodes:
            if  node['name']==link['source']:
                source_pos.append(node['pos'])
    

    #global source_pos_new

    for i in range(len(source_pos)):
        #遍历字典列表 for key,value in list_dict.items()并在key下的值做value*CANVAS_Y处理
        new_source={key:value*HEIGHT_SURFACE for key,value in source_pos[i].items()}
        source_pos_new.append( new_source)

    #取出target_pos
    global target_pos

    for link in all_links:
        for node in all_nodes:
            if  node['name']==link['target']:
               target_pos.append(node['pos'])


    #字典中的所有value都乘以CANVAS_Y
    #global target_pos_new

    for i in range(len(target_pos)):
        new_target={key:value*HEIGHT_SURFACE for key,value in target_pos[i].items()}
        target_pos_new.append(new_target)
    print("1source",len(source_pos_new))
    print("1target",len(target_pos_new))

    return{
        "ball1":source_pos_new,
        "ball2":target_pos_new
    }
   

distance_ST=[]
angle1s_radian=[]
def generate_stickyspike(drawStyle,encodeInfo,prop,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS):
    
    ss=getStickiness(encodeInfo,prop)
    global angle1s_radian,distance_ST

 
    #计算sourceball、targetball之间的连线与基准线的夹角
    for i in range(len(source_pos)):
        dy=target_pos_new[i]['y']-source_pos_new[i]['y']
        dx=target_pos_new[i]['x']-source_pos_new[i]['x']

        dxy=turtle.distance(dx,dy)
        distance_ST.append(dxy)
        if dx == 0:
            angle1_some=math.pi
        else:
            angle1_some=math.atan2(dy, dx)
            
        angle1s_radian.append(angle1_some)

    for item in distance_ST:
        if item<=abs(SOURCE_RADIUS-TARGET_RADIUS):
            print('\033[0;32;40m----------------------------两点重合--------------------------------\033[m')
            return
    #当两点重合时，重新计算source-target之间的半径
        d1=min(1,item*1/(SOURCE_RADIUS+TARGET_RADIUS))
        SOURCE_RADIUS*=d1
        TARGET_RADIUS*=d1
   
    anchors=getAnchors(source_pos_new,target_pos_new,ss,drawStyle,SOURCE_RADIUS,TARGET_RADIUS)

    #print(type(anchors))
    
    p1a=anchors['p1a']
    p1b=anchors['p1b']
    p2a=anchors['p2a']
    p2b=anchors['p2b']
   
    target_x=["x"]
    target_y=["y"]
    
    p1c=[]
    p1c=copy.deepcopy(source_pos_new)
    for i in range(len(p1c)):
        L1=SOURCE_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*ss
        x_c1i={key:value+L1*math.cos(angle1s_radian[i]) for key ,value in p1c[i].items() if key in target_x} 
        y_c1i={key:value+L1*math.sin(angle1s_radian[i])for key ,value in p1c[i].items() if key in target_y}
        p1c[i]['x']=x_c1i['x']
        p1c[i]['y']=y_c1i['y']

    p2c=[]
    p2c=copy.deepcopy(target_pos_new)
    for i in range(len(p2c)):
        L2=TARGET_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*ss
        x_c2i={key:value+L2*math.cos(angle1s_radian[i]+math.pi) for key ,value in p2c[i].items() if key in target_x} 
        y_c2i={key:value+L2*math.sin(angle1s_radian[i]+math.pi)for key ,value in p2c[i].items() if key in target_y}
        p2c[i]['x']=x_c2i['x']
        p2c[i]['y']=y_c2i['y']
    delta=0.8
    #stickeness<0.5，two spikes
    if(ss<=0.5):
    #spike for source
        

        if(drawStyle=='straight'):
        #spike1
            thirdpos1=copy.deepcopy(source_pos_new)
            for i in range(len(thirdpos1)):
                xi_thirdpos1={key:value+p1c[i]['x']-p1b[i]['x'] for key ,value in thirdpos1[i].items() if key in target_x}
                yi_thirdpos1={key:value+p1c[i]['y']-p1b[i]['y'] for key ,value in thirdpos1[i].items() if key in target_y}
                thirdpos1[i]['x']=xi_thirdpos1['x']
                thirdpos1[i]['y']=yi_thirdpos1['y']

            thirdpos2=copy.deepcopy(source_pos_new)
            for i in range(len(thirdpos2)):
                xi_thirdpos2={key:value+p1c[i]['x']-p1a[i]['x'] for key ,value in thirdpos2[i].items() if key in target_x}
                yi_thirdpos2={key:value+p1c[i]['y']-p1a[i]['y'] for key ,value in thirdpos2[i].items() if key in target_y}
                thirdpos2[i]['x']=xi_thirdpos2['x']
                thirdpos2[i]['y']=yi_thirdpos2['y']
            
        elif(drawStyle=='spike'):
        
           
            thirdpos_source=[]
            thirdpos_source=copy.deepcopy(p1c)
            
            handlepos1=[]
            handlepos1=copy.deepcopy(source_pos_new)
            for i in range(len(handlepos1)):
                L3=SOURCE_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*ss*0.1
                handlepos1_xi={key:value+L3*math.cos(angle1s_radian[i]) for key ,value in handlepos1[i].items() if key in target_x}
                handlepos1_yi={key:value+L3*math.sin(angle1s_radian[i]) for key,value in handlepos1[i].items() if key in target_y}
                handlepos1[i]['x']=handlepos1_xi['x']
                handlepos1[i]['y']=handlepos1_yi['y']
            
            p1a_handleIn=[]
            p1a_handleIn=copy.deepcopy(p1a)
            
            
            p1b_handleOut=[]
            p1b_handleOut=copy.deepcopy(p1b)
           
        else:
            print('convention')

     #spike2
        #global thirdpos3,thirdpos4

        if(drawStyle=='straight'):
            print('\033[0;32;40m----------------------------进入spike2下的straight风格---------------------------------\033[m')
            thirdpos3=copy.deepcopy(target_pos_new)
            for i in range(len(thirdpos3)):
                xi_thirdpos3={key:value+p2c[i]['x']-p2b[i]['x'] for key ,value in thirdpos3[i].items() if key in target_x}
                yi_thirdpos3={key:value+p2c[i]['y']-p2b[i]['y'] for key ,value in thirdpos3[i].items() if key in target_y}
                thirdpos3[i]['x']=xi_thirdpos3['x']
                thirdpos3[i]['y']=yi_thirdpos3['y']

            thirdpos4=copy.deepcopy(target_pos_new)
            for i in range(len(thirdpos4)):
                xi_thirdpos4={key:value+p2c[i]['x']-p2a[i]['x'] for key ,value in thirdpos4[i].items() if key in target_x}
                yi_thirdpos4={key:value+p2c[i]['y']-p2a[i]['y'] for key ,value in thirdpos4[i].items() if key in target_y}
                thirdpos4[i]['x']=xi_thirdpos4['x']
                thirdpos4[i]['y']=yi_thirdpos4['y']
            
            return{
            'thirdpos1':thirdpos1,
            'thirdpos2':thirdpos2,
            'thirdpos3':thirdpos3,
            'thirdpos4':thirdpos4
            }

        elif(drawStyle=='spike'):
            
            thirdpos_target=[]
            thirdpos_target=copy.deepcopy(p2c)
            
            handlepos2=[]
            handlepos2=copy.deepcopy(target_pos_new)
            for i in range(len(handlepos2)):
                L4=TARGET_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*ss*0.1
                handlepos2_xi={key:value+L4*math.cos(angle1s_radian[i]+math.pi) for key ,value in handlepos2[i].items() if key in target_x}
                handlepos2_yi={key:value+L4*math.sin(angle1s_radian[i]+math.pi) for key,value in handlepos2[i].items() if key in target_y}
                handlepos2[i]['x']=handlepos2_xi['x']
                handlepos2[i]['y']=handlepos2_yi['y']
            
            p2a_handleIn=[]
            
            
            p2b_handleOut=[]
            p2b_handleOut=copy.deepcopy(p2b)
           
            return{
                'thirdpos_source':thirdpos_source,
                'thirdpos_target':thirdpos_target,
                'p1a_handleIn':p1a_handleIn,
                'p1b_handleOut':p1b_handleOut,
                'p2a_handleIn':p2a_handleIn,
                'p2b_handleOut':p2b_handleOut
            }

        else:
            print('convention')
    else:
       
        thirdpos1_max=[]
        thirdpos1_max=copy.deepcopy(p1a)
        for i in range(len(thirdpos1_max)):
            xi_thirdpos1_max={key:(value+p2a[i]['x'])*0.5 for key ,value in thirdpos1_max[i].items() if key in target_x}
            yi_thirdpos1_max={key:(value+p2a[i]['y'])*0.5 for key ,value in thirdpos1_max[i].items() if key in target_y}
            thirdpos1_max[i]['x']=xi_thirdpos1_max['x']
            thirdpos1_max[i]['y']=yi_thirdpos1_max['y']
        thirdpos1_min=[]
        thirdpos1_min=copy.deepcopy(source_pos_new)
        for i in range(len(thirdpos1_min)):
            L5=SOURCE_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*0.5
            xi_thirdpos1_min={key:value+L5*math.cos(angle1s_radian[i]) for key ,value in thirdpos1_min[i].items() if key in target_x}
            yi_thirdpos1_min={key:value+L5*math.sin(angle1s_radian[i]) for key ,value in thirdpos1_min[i].items() if key in target_y}
            thirdpos1_min[i]['x']=xi_thirdpos1_min['x']
            thirdpos1_min[i]['y']=yi_thirdpos1_min['y']
        thirdpos1= interpolateV(thirdpos1_min,thirdpos1_max,(ss-0.5)/0.5)
       
       
        thirdpos2_max=[]
        thirdpos2_max=copy.deepcopy(p1b)
        for i in range(len(thirdpos2_max)):
            xi_thirdpos2_max={key:(value+p2b[i]['x'])*0.5 for key,value in thirdpos2_max[i].items() if key in target_x}
            yi_thirdpos2_max={key:(value+p2b[i]['y'])*0.5 for key,value in thirdpos2_max[i].items() if key in target_y}
            thirdpos2_max[i]['x']=xi_thirdpos2_max['x']
            thirdpos2_max[i]['y']=yi_thirdpos2_max['y']
        thirdpos2_min=[]
        thirdpos2_min=copy.deepcopy(target_pos_new)
        for i in range(len(thirdpos2_min)):
            L6=TARGET_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*0.5
            xi_thirdpos2_min={key:value+L6*math.cos(angle1s_radian[i]+math.pi) for key ,value in thirdpos2_min[i].items() if key in target_x}
            yi_thirdpos2_min={key:value+L6*math.sin(angle1s_radian[i]+math.pi) for key ,value in thirdpos2_min[i].items() if key in target_y}
            thirdpos2_min[i]['x']=xi_thirdpos2_min['x']
            thirdpos2_min[i]['y']=yi_thirdpos2_min['y']
        thirdpos2= interpolateV(thirdpos2_min,thirdpos2_max,(ss-0.5)/0.5)

        if(drawStyle=='straight'):
            return {
                'p1a':p1a,
                'p2a':p2a,
                'p2b':p2b,
                'p1b':p1b
            }
        
        else:
            print('-----------------------进入0.8下的spike参数计算部分-------------------------------')
            
            ##interR=math.pow((ss-0.5)/0.5,3)
            interR=0.5
            midpos1_base,midpos2_base=[],[]

            v_length1,v_length2,v_length3,v_length4=[],[],[],[]
            midpos1_base=copy.deepcopy(source_pos_new)
            midpos2_base=copy.deepcopy(target_pos_new)
            for i in range(len(midpos1_base)):
                L7=SOURCE_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*0.5*0.1
                x_midpos1_base={key:value+L7*math.cos(angle1s_radian[i]) for key ,value in midpos1_base[i].items() if key in target_x} 
                y_midpos1_base={key:value+L7*math.sin(angle1s_radian[i]) for key ,value in midpos1_base[i].items() if key in target_y}
                midpos1_base[i]['x']=x_midpos1_base['x']
                midpos1_base[i]['y']=y_midpos1_base['y']

                v_length1.append(getVectorLength(
                {
                    'x':(x_midpos1_base['x']-p1a[i]['x']),
                    'y':(y_midpos1_base['y']-p1a[i]['y'])
                }))
                
                v_length2.append(getVectorLength(
                {
                    'x':(x_midpos1_base['x']-p1b[i]['x']),
                    'y':(y_midpos1_base['y']-p1b[i]['y'])
                }))
                
                L8=TARGET_RADIUS+(distance_ST[i]-(SOURCE_RADIUS+TARGET_RADIUS))*0.5*0.1
                x_midpos2_base={key:value+L8*math.cos(angle1s_radian[i]-math.pi) for key ,value in midpos2_base[i].items() if key in target_x}
                y_midpos2_base={key:value+L8*math.sin(angle1s_radian[i]-math.pi) for key ,value in midpos2_base[i].items() if key in target_y}
                midpos2_base[i]['x']=x_midpos2_base['x']
                midpos2_base[i]['y']=y_midpos2_base['y']

                v_length3.append(getVectorLength(
                {
                    'x':(x_midpos2_base['x']-p2a[i]['x']),
                    'y':(y_midpos2_base['y']-p2a[i]['y'])
                }))

                v_length4.append(getVectorLength(
                {
                    'x':(x_midpos2_base['x']-p2b[i]['x']),
                    'y':(y_midpos2_base['y']-p2b[i]['y'])
                }))
           

            


            h3=interpolateVS(p1b,midpos1_base,p2b,interR,v_length2)
            h1=interpolateVS(p2a,midpos2_base,p1a,interR,v_length3)
            h2=interpolateVS(p2b,midpos2_base,p1b,interR,v_length4)

            p1a_handleOut=[]
            p1a_handleOut=copy.deepcopy(h0)
            for i in range(len(p1a_handleOut)):
                p1a_handleOut_xi={key:(value+source_pos_new[i]['x'])*delta for key ,value in  p1a_handleOut[i].items() if key in target_x}
                p1a_handleOut_yi={key:(value+source_pos_new[i]['y'])*delta for key ,value in  p1a_handleOut[i].items() if key in target_y}
                p1a_handleOut[i]['x']=p1a_handleOut_xi['x']
                p1a_handleOut[i]['y']=p1a_handleOut_yi['y']
            print('\033[0;32;40m---------------------------p1a_handleOut---------------------------------\033[m')
            print(p1a_handleOut[0:10])
            

            p2a_handleIn=[]
            p2a_handleIn=copy.deepcopy(h1)
            for i in range(len(p2a_handleIn)):
                p2a_handleIn_xi={key:(value+source_pos_new[i]['x'])*delta for key ,value in  p2a_handleIn[i].items() if key in target_x}
                p2a_handleIn_yi={key:(value+source_pos_new[i]['y'])*delta for key ,value in  p2a_handleIn[i].items() if key in target_y}
                p2a_handleIn[i]['x']=p2a_handleIn_xi['x']
                p2a_handleIn[i]['y']=p2a_handleIn_yi['y']

            p2b_handleOut=[]
            p2b_handleOut=copy.deepcopy(h2)
            for i in range(len(p2b_handleOut)):
                p2b_handleOut_xi={key:(value+source_pos_new[i]['x'])*delta for key ,value in  p2b_handleOut[i].items() if key in target_x}
                p2b_handleOut_yi={key:(value+source_pos_new[i]['y'])*delta for key ,value in  p2b_handleOut[i].items() if key in target_y}
                p2b_handleOut[i]['x']=p2b_handleOut_xi['x']
                p2b_handleOut[i]['y']=p2b_handleOut_yi['y']
            
            p1b_handleIn=[]
            p1b_handleIn=copy.deepcopy(h3)
            for i in range(len(p1b_handleIn)):
                p1b_handleIn_xi={key:(value+source_pos_new[i]['x'])*delta for key, value in p1b_handleIn[i].items() if key in target_x}
                p1b_handleIn_yi={key:(value+source_pos_new[i]['y'])*delta for key, value in p1b_handleIn[i].items() if key in target_y}
                p1b_handleIn[i]['x']=p1b_handleIn_xi['x']
                p1b_handleIn[i]['y']=p1b_handleIn_yi['y']
            
            return {
                'thirdpos1':thirdpos1,
                'thirdpos2':thirdpos2,
                'p1a_handleOut':p1a_handleOut,
                'p2a_handleIn':p2a_handleIn,
                'p2b_handleOut':p2b_handleOut,
                'p1b_handleIn':p1b_handleIn
            }
def getStickiness(encodeInfo,prop):
    if (encodeInfo=='constant'):
        stickiness=prop['constant']
    return stickiness

def getAnchors(ball1_pos,ball2_pos,stickiness,drawStyle,SOURCE_RADIUS,TARGET_RADIUS):
    global u1,u2
    maxAngDif=math.pi*0.35
    minAngDif=math.pi*0.05
    angleDif=minAngDif+(maxAngDif-minAngDif)*stickiness

    dxy=copy.deepcopy(distance_ST)
    for i in range(len(dxy)):
        if(dxy[i]<SOURCE_RADIUS+TARGET_RADIUS):
            u1=math.math.acos((SOURCE_RADIUS*SOURCE_RADIUS+dxy*dxy*TARGET_RADIUS*TARGET_RADIUS))/(2*SOURCE_RADIUS*dxy)
            u2=math.math.acos((TARGET_RADIUS*TARGET_RADIUS+dxy*dxy-SOURCE_RADIUS*SOURCE_RADIUS))/(2*TARGET_RADIUS*dxy)
        else:
            u1=0
            u2=0
    
    if(drawStyle=='straight'):
        angleDif=minAngDif+(maxAngDif-minAngDif)*0.1

    angle1a=[]
    global angle1s_radian
   
    for i in range(len(angle1s_radian)):
        a1=angle1s_radian[i]+angleDif
        angle1a.append(a1)
   

    angle1b=[]
    for i in range(len(angle1s_radian)):
        b1=angle1s_radian[i]-angleDif
        angle1b.append(b1)
    

    angle2a=[]
    for i in range(len(angle1s_radian)):
        a2=angle1s_radian[i]+math.pi-angleDif
        angle2a.append(a2)
   

    angle2b=[]
    for i in range(len(angle1s_radian)):
        b2=angle1s_radian[i]-math.pi+angleDif
        angle2b.append(b2)
   
    
  
    target_x=["x"]
    target_y=["y"]
   
    p1a=[]
    p1a=copy.deepcopy(ball1_pos)
   
    
    for i in range(len(p1a)):
        p1a_xi={key:value+SOURCE_RADIUS*math.cos(angle1a[i]) for key ,value in p1a[i].items() if key in target_x} 
        p1a_yi={key:value+SOURCE_RADIUS*math.sin(angle1a[i]) for key ,value in p1a[i].items() if key in target_y}
        p1a[i]['x']=p1a_xi['x']
        p1a[i]['y']=p1a_yi['y']

    
    p1b=[]
    p1b=copy.deepcopy(ball1_pos)
    for i in range(len(p1b)):
        p1b_xi={key:value+SOURCE_RADIUS*math.cos(angle1b[i]) for key ,value in p1b[i].items() if key in target_x} 
        p1b_yi={key:value+SOURCE_RADIUS*math.sin(angle1b[i]) for key ,value in p1b[i].items() if key in target_y}
        p1b[i]['x']=p1b_xi['x']
        p1b[i]['y']=p1b_yi['y']


    p2a=[]
    p2a=copy.deepcopy(ball2_pos)
    for i in range(len(p2a)):
        p2a_xi={key:value+TARGET_RADIUS*math.cos(angle2a[i]) for key ,value in p2a[i].items() if key in target_x} 
        p2a_yi={key:value+TARGET_RADIUS*math.sin(angle2a[i]) for key ,value in p2a[i].items() if key in target_y}
        p2a[i]['x']=p2a_xi['x']
        p2a[i]['y']=p2a_yi['y']
    
  
    p2b=[]
    p2b=copy.deepcopy(ball2_pos)
    for i in range(len(p2b)):
        p2b_xi={key:value+TARGET_RADIUS*math.cos(angle2b[i]) for key ,value in p2b[i].items() if key in target_x} 
        p2b_yi={key:value+TARGET_RADIUS*math.sin(angle2b[i]) for key ,value in p2b[i].items() if key in target_y}
        p2b[i]['x']=p2b_xi['x']
        p2b[i]['y']=p2b_yi['y']

  
    return {
		'p1a': p1a,
		'p1b': p1b,
		'p2a': p2a,
		'p2b': p2b
	}


def drawStyle(stickiness,pab,thirds,drawStyle,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS,WIDTH_SURFACE,HEIGHT_SURFACE):

    p1a_x,p1a_y=[],[]
    p1b_x,p1b_y=[],[]
    p2a_x,p2a_y=[],[]
    p2b_x,p2b_y=[],[]
    for item in pab:
        for j in pab[item]:
            if(item=='p1a'):
                p1a_x.append(j['x'])
                p1a_y.append(j['y'])
            elif(item=='p1b'):
                    p1b_x.append(j['x'])
                    p1b_y.append(j['y'])
            elif(item=='p2a'):
                    p2a_x.append(j['x'])
                    p2a_y.append(j['y'])
            else:
                    p2b_x.append(j['x'])
                    p2b_y.append(j['y'])
    if(stickiness<=0.5 and drawStyle=='straight'):
        
        thirdpos1_x,thirdpos1_y=[],[]
        thirdpos2_x,thirdpos2_y=[],[]
        thirdpos3_x,thirdpos3_y=[],[]
        thirdpos4_x,thirdpos4_y=[],[]

        for ele in thirds:
            for k in thirds[ele]:
                if(ele=='thirdpos1'):
                    thirdpos1_x.append(k['x'])
                    thirdpos1_y.append(k['y'])
                elif(ele=='thirdpos2'):
                    thirdpos2_x.append(k['x'])
                    thirdpos2_y.append(k['y'])
                elif(ele=='thirdpos3'):
                    thirdpos3_x.append(k['x'])
                    thirdpos3_y.append(k['y'])
                else:
                    thirdpos4_x.append(k['x'])
                    thirdpos4_y.append(k['y'])
        with cairo.SVGSurface( 'straight.svg',WIDTH_SURFACE,HEIGHT_SURFACE) as surface:
            context=cairo.Context(surface)

            #画矩形边
            for i in range(len(p1a_x)):
                context.set_source_rgb(0.5, 0.5, 0.5)
                context.set_line_width(1)

                context.move_to(p1a_x[i],p1a_y[i])
                context.set_line_cap(cairo.LINE_CAP_BUTT)
                context.line_to(thirdpos1_x[i],thirdpos1_y[i])
                context.line_to(thirdpos2_x[i],thirdpos2_y[i])
                context.line_to(p1b_x[i],p1b_y[i])
                context.line_to(p1a_x[i],p1a_y[i])
                context.fill()
                context.stroke()


                context.move_to(p2a_x[i],p2a_y[i])
                context.set_line_cap(cairo.LINE_CAP_BUTT)
                context.line_to(thirdpos3_x[i],thirdpos3_y[i])
                context.line_to(thirdpos4_x[i],thirdpos4_y[i])
                context.line_to(p2b_x[i],p2b_y[i])
                context.line_to(p2a_x[i],p2a_y[i])
                context.fill()
                context.stroke()
            #画⚪
            drawCircles(context,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS)

    elif(stickiness<=0.5 and drawStyle=='spike'):
        p1a_handle=[]
        p1b_handle=[]
        thirdpos_source=[]
        thirdpos_target=[]
        p2a_handle=[]
        p2b_handle=[]
        for item in thirds:
            for i in thirds[item]:
                if(item=='thirdpos_source'):
                    thirdpos_source=copy.deepcopy(thirds[item])
                elif(item=='thirdpos_target'):
                    thirdpos_target=copy.deepcopy(thirds[item])
                elif(item=='p1a_handleIn'):
                    p1a_handle=copy.deepcopy(thirds[item])
                elif(item=='p1b_handleOut'):
                    p1b_handle=copy.deepcopy(thirds[item])
                elif(item=='p2a_handleIn'):
                    p2a_handle=copy.deepcopy(thirds[item])
                else:
                    p2b_handle=copy.deepcopy(thirds[item])

        with cairo.SVGSurface('spike_seperated.svg',WIDTH_SURFACE,HEIGHT_SURFACE) as surface:
            context = cairo.Context(surface)
    # move the context to x,y position
            for i in range (len(p1a_x)):
    # Drawing Curve
                context.set_line_width(2.12)
                context.set_source_rgb(0.5, 0.5, 0.5)

                #context.move_to(p1a_x[i],p1a_y[i])
                context.curve_to(p1a_x[i],p1a_y[i],p1a_handle[i]['x'],p1a_handle[i]['y'],thirdpos_source[i]['x'],thirdpos_source[i]['y'])
                context.curve_to(thirdpos_source[i]['x'],thirdpos_source[i]['y'],p1b_handle[i]['x'],p1b_handle[i]['y'],p1b_x[i],p1b_y[i])
                context.set_line_cap(cairo.LINE_CAP_BUTT)
                #context.line_to(p1a_x[i],p1a_y[i])
                context.fill()


                #context.move_to(p2a_x[i],p2a_y[i])
                context.curve_to(p2a_x[i],p2a_y[i],p2a_handle[i]['x'],p2a_handle[i]['y'],thirdpos_target[i]['x'],thirdpos_target[i]['y'])
                context.curve_to(thirdpos_target[i]['x'],thirdpos_target[i]['y'],p2b_handle[i]['x'],p2b_handle[i]['y'],p2b_x[i],p2b_y[i])
                context.set_line_cap(cairo.LINE_CAP_BUTT)
                #context.line_to(p2a_x[i],p2a_y[i])
                context.fill()
            drawCircles(context,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS)

    elif(stickiness<=0.5 and drawStyle=='convention'):

        rela=relationMatrix()

        [rows_ST_pos,cols_ST_pos]=rela.shape
        with cairo.SVGSurface( 'convention.svg',WIDTH_SURFACE,HEIGHT_SURFACE) as surface:
            for source in range(rows_ST_pos):
                for target in range(cols_ST_pos):
        #取出第1colume
                    if target==0:
                        ele_source=rela[source,target]
                        ele_target=rela[source,target+1]
        #取出同一行的第2colume
                    else:
                        ele_source=rela[source,target-1]
                        ele_target=rela[source,target]
        #画边
                    context=cairo.Context(surface)
                    context.set_source_rgb(0.5, 0.5, 0.5)
                    context.set_line_width(2.12)

                    context.move_to(all_pos[ele_source]['x']*HEIGHT_SURFACE,all_pos[ele_source]['y']*HEIGHT_SURFACE)
                    context.set_line_cap(cairo.LINE_CAP_BUTT)
                    context.line_to(all_pos[ele_target]['x']*HEIGHT_SURFACE,all_pos[ele_target]['y']*HEIGHT_SURFACE)
                    context.stroke()
        #画⚪
            drawCircles(context,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS)

    elif (stickiness>0.5 and drawStyle=='straight'):
        pass
    
def relationMatrix():
    SourceTargets_matrix=np.zeros((len(all_nodes),len(all_nodes)),dtype=int)
    all_names_dictkey=[]
    right_value=len(all_names)+1
    for i in range(0,right_value):
        all_names_dictkey.append(i)

    all_namesid_lst=zip(all_names,all_names_dictkey)

    all_namesid=dict(all_namesid_lst)
    for item in source_target_allout:
        row=all_namesid[item["target"]]
        colume=all_namesid[item["source"]]
        SourceTargets_matrix[row][colume]=1
    two_awayposition=SourceTargets_matrix.nonzero()
    ST_pos=np.column_stack(two_awayposition)
    print(type(ST_pos))
    return ST_pos


def drawCircles(context,center1,center2,radius1,radius2):
    for i in range (len(center1)):
        context.arc(center1[i]['x'],center1[i]['y'],radius1,0,2*math.pi)
        context.arc(center2[i]['x'],center2[i]['y'],radius2,0,2*math.pi)
        context.set_source_rgb(0, 0, 0) #填充圆的颜色
        context.fill()

def getVectorLength(v):
    '''
    length=[]
    for i in range(len(v)):
        length.append(math.sqrt(v[i]['x']*v[i]['x']+v[i]['y']*v[i]['y']))
    return length
    '''
    return math.sqrt(v['x']*v['x']+v['y']*v['y'])

    #h0=interpolateVS(p1a,midpos1_base,p2a,interR,v_length1)
def interpolateVS(centerP,fromP,toP,ratio,radius):
    target_x=["x"]
    target_y=["y"]
    #v1,v2,v=[],[],[]
    v1=copy.deepcopy(fromP)
    v2=copy.deepcopy(toP)
    for i in range (len(v1)):
        xi_v1={key:value-centerP[i]['x'] for key ,value in v1[i].items() if key in target_x} 
        yi_v1={key:value-centerP[i]['y'] for key ,value in v1[i].items() if key in target_y}
        v1[i]['x']=xi_v1['x']
        v1[i]['y']=yi_v1['y']
    for i in range (len(v2)):
        xi_v2={key:value-centerP[i]['x'] for key ,value in v2[i].items() if key in target_x} 
        yi_v2={key:value-centerP[i]['y'] for key ,value in v2[i].items() if key in target_y}
        v2[i]['x']=xi_v2['x']
        v2[i]['y']=yi_v2['y']
    
    
    v= interpolateV(v1,v2,ratio)

    for i in range(len(v)):
        item=radius[i]/ getVectorLength(v[i])
        #print('item',item)
        xi_v={key:value*item for key ,value in v[i].items() if key in target_x} 
        yi_v={key:value*item for key ,value in v[i].items() if key in target_y}
        v[i]['x']=xi_v['x']
        v[i]['y']=yi_v['y']
    return v
       

#thirdpos1= interpolateV(thirdpos1_min,thirdpos1_max,(ss-0.5)/0.5)
def interpolateV(v1,v2,ratio):
    target_x=["x"]
    target_y=["y"]
    for i in range(len(v1)):
        
        xi_v1={key:value+ratio*(v2[i]['x']-value) for key ,value in v1[i].items() if key in target_x} 
        yi_v1={key:value+ratio*(v2[i]['y']-value) for key ,value in v1[i].items() if key in target_y}
        v1[i]['x']=xi_v1['x']
        v1[i]['y']=yi_v1['y']
    return v1
        

if __name__ == '__main__':
    WIDTH_SURFACE=700
    HEIGHT_SURFACE=700
    TARGET_RADIUS=5
    SOURCE_RADIUS=5
    encodeInfo='constant'
    prop={
         'constant':0.8
    }

    my_drawStyle=input('绘制风格(straight/spike/convention):')
    path=('C:\\FilesSelf\\SS05\\miserables_layout.json')
    readdata(path)
    nodes_out()
    links_out()
    
  
    ss_spe=getStickiness(encodeInfo,prop)
    position=get_sourcetargetpos(HEIGHT_SURFACE)
    
    source_pos_new=position.get('ball1')
    target_pos_new=position.get('ball2')
    
    thirds_handles=generate_stickyspike(my_drawStyle,encodeInfo,prop,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS)

    pab=getAnchors(source_pos_new,target_pos_new,ss_spe,my_drawStyle,SOURCE_RADIUS,TARGET_RADIUS)

    drawStyle(ss_spe,pab,thirds_handles,my_drawStyle,source_pos_new,target_pos_new,SOURCE_RADIUS,TARGET_RADIUS,WIDTH_SURFACE,HEIGHT_SURFACE) 

    
