//
// Created by lj on 2022/11/6.
//

#ifndef FUXIAN_SAMPLE_POINT_H
#define FUXIAN_SAMPLE_POINT_H


#include "point.h"

class SamplePoint: public Point{
public:
    float quality=0;
    float mark=0;  //可见性标记
    SamplePoint();

    SamplePoint(float x, float y, float z, float nx, float ny, float nz, float quality);
    SamplePoint(float x, float y, float z, float nx, float ny, float nz);

    SamplePoint(float *pos, float *direction);

//    SamplePoint(Point &&point);
    // 拷贝构造函数
    SamplePoint(const SamplePoint& samplePoint);

    // 重写父类方法
    float* getPos();
    float* getDirection();
    void printInfo();

    //重载运算符=
    SamplePoint& operator=(SamplePoint &sp);
};


#endif //FUXIAN_SAMPLE_POINT_H
