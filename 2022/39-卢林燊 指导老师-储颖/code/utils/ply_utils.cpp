//
// Created by lj on 2022/11/9.
//

#include "ply_utils.h"
#include <math.h>

#define M_PI 3.1415926


//计算弧度
float PlyUtils::calRadianOfPoints(const float *d1, const float *d2) {
    //  d1 {x,y,0}  d2 {x,y,z}
    float product = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];  //x*x+y*y
    float len1 = sqrt(pow(d1[0], 2) + pow(d1[1], 2) + pow(d1[2], 2));
    float len2 = sqrt(pow(d2[0], 2) + pow(d2[1], 2) + pow(d2[2], 2));
    float result = product / (len1 * len2);
    if(result > 1){
        result = 1;
    }else if(result < -1){
        result = -1;
    }
    return acos(result);
}

float PlyUtils::radian2angle(float radian) {
    return radian / M_PI * 180;  //  1弧度=(180/π)°
}


