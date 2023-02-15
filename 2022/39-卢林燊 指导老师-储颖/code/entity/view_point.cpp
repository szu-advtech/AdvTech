//
// Created by lj on 2022/11/6.
//

#include <iostream>
#include "view_point.h"
#include "pos_pack.h"
#include "../utils/tools.h"

ViewPoint::ViewPoint() = default;

ViewPoint::ViewPoint(float *pos, float *direction): Point(pos, direction) {
    this->buildCameraMatrix(pos, direction);
}

ViewPoint::ViewPoint(float x, float y, float z, float nx, float ny, float nz): Point(x, y, z, nx, ny, nz) {
    float pos[3] = {x, y, z};
    float direction[3] = {nx, ny, nz};
    this->buildCameraMatrix(pos, direction);
}

// 拷贝构造函数
ViewPoint::ViewPoint(const ViewPoint &viewPoint): Point(viewPoint) {
    this->cameraMatrix = viewPoint.cameraMatrix;
    this->angle = viewPoint.angle;
    this->score = viewPoint.score;
}

const float * ViewPoint::getPos() {
    return this->_pos;
}

const float * ViewPoint::getDirection() {
    return this->_direction;
}


void ViewPoint::setPos(float *pos) {
    for(int i=0; i<3; i++){
        this->_pos[i] = pos[i];
    }
    this->buildCameraMatrix(this->_pos, this->_direction);
}

void ViewPoint::setPosByIndex(int index, float val) {
    this->_pos[index] = val;
    this->buildCameraMatrix(this->_pos, this->_direction);
}

void ViewPoint::setDirectionByIndex(int index, float val) {
    this->_direction[index] = val;
    this->buildCameraMatrix(this->_pos, this->_direction);
}

void ViewPoint::setDirection(float *direction){
    for(int i=0; i<3; i++){
        this->_direction[i] = direction[i];
    }
    this->buildCameraMatrix(this->_pos, this->_direction);
}


void ViewPoint::buildCameraMatrix(float *pos, float *direction) {
    PosPack posPack = Tools::getPosPackFromDirectionVector(pos, direction);
    this->cameraMatrix = posPack.cameraMatrix;
}


void ViewPoint::printInfo() {
    std::cout << "viewPoint info: address:" << &this->_pos << "pos[" << this->_pos[0] << ", " << this->_pos[1] << ", " << this->_pos[2]
              << "], address:" << &this->_direction << "direction[" << this->_direction[0] << ", " << this->_direction[1] << ", " << this->_direction[2] << "]" << std::endl;
}


ViewPoint &ViewPoint::operator=(ViewPoint &viewPoint) {
    this->setPos(viewPoint._pos);
    this->setDirection(viewPoint._direction);
    this->angle = viewPoint.angle;
    this->cameraMatrix = viewPoint.cameraMatrix;
    this->angle = viewPoint.angle;
    this->score = viewPoint.score;
    return *this;
}

ViewPoint& ViewPoint::operator=(ViewPoint &&viewPoint) noexcept{
    this->_pos = viewPoint._pos;
    this->_direction = viewPoint._direction;
    viewPoint._pos = nullptr;
    viewPoint._direction = nullptr;
    this->angle = viewPoint.angle;
    this->cameraMatrix = viewPoint.cameraMatrix;
    this->angle = viewPoint.angle;
    this->score = viewPoint.score;
    return *this;
}