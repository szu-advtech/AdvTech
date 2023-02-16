//
// Created by lj on 2022/11/6.
//

#ifndef FUXIAN_VIEW_POINT_H
#define FUXIAN_VIEW_POINT_H

#include <Eigen/Geometry>
#include "point.h"

class ViewPoint : private Point{
public:
    Eigen::Isometry3f cameraMatrix;


    float angle = -1; // 视点和采样点法线的角度
    float score = 0;
    int selected = 0;

    ViewPoint();
    ViewPoint(float *pos, float *direction);
    ViewPoint(float x, float y, float z, float nx, float ny, float nz);
    // 拷贝构造函数
    ViewPoint(const ViewPoint& viewPoint);

    const float* getPos();
    const float* getDirection();
    void setPos(float *pos);
    void setPosByIndex(int index, float val);
    void setDirectionByIndex(int index, float val);
    void setDirection(float *direction);
    void printInfo();
    void buildCameraMatrix(float *pos, float *direction);


    ViewPoint& operator=(ViewPoint& viewPoint);
    ViewPoint& operator=(ViewPoint&& viewPoint) noexcept ;


};


#endif //FUXIAN_VIEW_POINT_H
