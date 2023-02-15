//
// Created by lj on 2022/11/6.
//

#include "point.h"
Point::Point() {
    this->_pos = new float[3];
    this->_direction = new float[3];
}

Point::Point(float x, float y, float z, float nx, float ny, float nz) {
    this->_pos = new float[3]{x, y, z};
    this->_direction = new float[3]{nx, ny, nz};
}

Point::Point(float *pos, float *direction) {
    this->_pos = new float[3]{pos[0], pos[1], pos[2]};
    this->_direction = new float[3]{direction[0], direction[1], direction[2]};
}

// 拷贝构造函数：深拷贝
Point::Point(const Point &point) {
    this->_pos = new float[3]{point._pos[0], point._pos[1], point._pos[2]};
    this->_direction = new float[3]{point._direction[0], point._direction[1], point._direction[2]};
}

// 移动构造函数
Point::Point(Point &&point) noexcept{
    this->_pos = point._pos;
    this->_direction = point._direction;
    point._pos = nullptr;
    point._direction = nullptr;
}

Point::~Point() {
    delete [] this->_pos;
    delete [] this->_direction;
}

float *Point::getPos() {
    return this->_pos;
}

float *Point::getDirection() {
    return this->_direction;
}