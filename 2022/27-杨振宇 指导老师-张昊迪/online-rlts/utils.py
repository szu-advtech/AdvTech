import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_loader import *

'''
sed:同步欧几里得距离
输入：一段轨迹段
过程：对从第二个到倒数第二个依次计算距离，求出最大的误差
输出：最大点的那个误差
'''
def sed(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        #print('segment', segment)
        point_start = segment[0]        #线段的第一个值
        point_end = segment[-1]       #线段的最后一个值
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][2]
            time_ratio = 1 if (point_end[2]- point_start[2]) == 0  else (syn_time-point_start[2]) / (point_end[2]-point_start[2])
            syn_x = point_start[0] + (point_end[0] - point_start[0]) * time_ratio
            syn_y = point_start[1] + (point_end[1] - point_start[1]) * time_ratio
            e = max(e, np.linalg.norm(np.array([segment[i][0],segment[i][1]]) - np.array([syn_x,syn_y])))
        #print('segment error', e)
        return e
'''
函数sed计算一个确定轨迹的误差
函数sed_max_e计算原始轨迹和简化轨迹之间的误差，即论文中谈及的误差
输入：原始轨迹（简化轨迹中间的所有点）和简化轨迹（两个相邻点）
输出：最大sed误差
'''
def sed_max_e(original_T, simplified_T):
    #original_T, simplified_T = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(original_T))]
    for c, value in enumerate(original_T):
        dict_traj[tuple(value)] = c
    for value in simplified_T:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, sed(original_T[start: c + 1]))
            start = c
    return t_map, error

def ped(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            pm = segment[i]
            A = pe[1] - ps[1]
            B = ps[0] - pe[0]
            C = pe[0] * ps[1] - ps[0] * pe[1]
            if A == 0 and B == 0:
                e = max(e, 0.0)
            else:
                e = max(e, abs((A * pm[0] + B * pm[1] + C)/ np.sqrt(A * A + B * B)))
        #print('segment error', e)
        return e

def ped_max_e(original_T, simplified_T):
    #original_T, simplified_T = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(original_T))]
    for c, value in enumerate(original_T):
        dict_traj[tuple(value)] = c
    for value in simplified_T:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, ped(original_T[start: c + 1]))
            start = c
    return t_map, error

def angle(v1):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    angle1 = math.atan2(dy1, dx1)
    if angle1 >= 0:
        return angle1
    else:
        return 2*math.pi + angle1

def dad_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        theta_0 = angle([ps[0],ps[1],pe[0],pe[1]])
        for i in range(0,len(segment)-1):
            pm_0 = segment[i]
            pm_1 = segment[i+1]
            theta_1 = angle([pm_0[0],pm_0[1],pm_1[0],pm_1[1]])
            e = max(e, min(abs(theta_0 - theta_1), 2*math.pi - abs(theta_0 - theta_1)))
        #print('segment error', e)
        return e

def dad_error(ori_traj, sim_traj):
    #original_T, simplified_T = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, dad_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def get_point(ps, pe, segment, index):
    syn_time = segment[index][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    return [syn_x, syn_y], syn_time

def speed_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(0,len(segment)-1):
            p_1, t_1 = get_point(ps, pe, segment, i)
            p_2, t_2 = get_point(ps, pe, segment, i+1)
            time = 1 if t_2 - t_1 == 0 else abs(t_2-t_1)
            est_speed = np.linalg.norm(np.array(p_1) - np.array(p_2))/time
            rea_speed = np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([segment[i+1][0], segment[i+1][1]]))/time
            e = max(e, abs(est_speed - rea_speed))
        #print('segment error', e)
        return e

def speed_error(ori_traj, sim_traj):
    #original_T, simplified_T = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, speed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

'''
接受原始轨迹和简化后的轨迹作为参数，并返回最大垂线距离和相关信息
'''
def draw_sed_op(segment):
    if len(segment) <= 2:
        # print('segment error', 0.0)
        return 0.0, segment[0], segment[0], segment[0], segment[0]
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1, len(segment) - 1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2] - ps[2]) == 0 else (syn_time - ps[2]) / (pe[2] - ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            t = np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([syn_x, syn_y]))
            if t >= e:
                e = t
                e_points = segment[i]
                syn = [syn_x, syn_y]
        print('segment error', e)
        return e, e_points, ps, pe, syn

'''
为了计算最大垂线距离，函数遍历原始轨迹中的所有折线段，
并调用draw_sed_op函数来计算每个折线段的最大垂线距离。最后，函数返回最大垂线距离和相关信息
'''
def draw_error(ori_traj, sim_traj, label):
    # original_T, simplified_T = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            # print(start, c)
            if label == 'sed':
                e, e_points, ps, pe, syn = draw_sed_op(ori_traj[start: c + 1])
                if e > error:
                    error = e
                    error_points = e_points
                    error_syn = syn
                    error_left = ps
                    error_right = pe
            start = c
    return error, error_points, error_left, error_right, error_syn

'''
draw函数则接受原始轨迹和简化后的轨迹作为参数，
并使用这些轨迹信息绘制图像。为了获取最大垂线距离和相关信息，
函数调用draw_error函数。然后使用最大垂线距离信息在图像中绘制垂线，
并使用其他信息绘制轨迹图
'''
def draw(ori_traj, sim_traj, label='sed'):
    error, error_points, error_left, error_right, error_syn = draw_error(ori_traj, sim_traj, label)
    pdf = PdfPages('vis_rlts_geo_sed_online.pdf')
    plt.figure(figsize=(10.5 / 2, 6.8 / 2))
    plt.plot(np.array(ori_traj)[:, 0], np.array(ori_traj)[:, 1], color="blue", linewidth=0.7, label='raw traj')
    plt.scatter(np.array(sim_traj)[:, 0], np.array(sim_traj)[:, 1], color="red", s=20)
    plt.plot(np.array(sim_traj)[:, 0], np.array(sim_traj)[:, 1], '--', color="red", linewidth=0.5,
             label='simplified traj')
    # plt.scatter(error_points[0],error_points[1],color="black", s=30, marker='s', label='maximal error point')
    plt.plot([error_points[0], error_syn[0]], [error_points[1], error_syn[1]], '--', color="black", label='SED')
    plt.plot([error_left[0], error_right[0]], [error_left[1], error_right[1]], color="green", linewidth=2,
             label='anchor seg')
    plt.title('simplified traj length: ' + str(len(sim_traj)))
    plt.legend(loc='best', prop={'size': 9})
    pdf.savefig()
    pdf.close()
    return error

if __name__ == '__main__':
    print('测试，无语法错误!')
