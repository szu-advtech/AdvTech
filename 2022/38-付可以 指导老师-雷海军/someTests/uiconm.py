import cv2
import math
import numpy as np

def UICONM(rbg, L):  #wrong
    m, n, o = np.shape(rbg)  #横向为n列 纵向为m行
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    A1 = np.zeros((L, L)) #全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max+rbg_min-rbg_max*rbg_min/1026
            if 1026-rbg_min > 0:
                plip_del = 1026*(rbg_max-rbg_min)/(1026-rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del/plip_add
                    local_b = math.log(plip_del/plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026-1026*((1-logAMEE/1026)**(1/(number_n*number_m)))
    return logAMEE


def fky_uiconm(img_path):
    img = cv2.imread(img_path)
    Uiconm = UICONM(img, 8)
    print(Uiconm)

fky_uiconm("WBP-test1.png")
fky_uiconm("WBP-test2.png")
fky_uiconm("WBP-test3.png")
fky_uiconm("WBP-test4.png")
fky_uiconm("WBP-test5.png")