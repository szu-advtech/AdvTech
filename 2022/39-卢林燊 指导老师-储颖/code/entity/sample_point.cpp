//
// Created by lj on 2022/11/6.
//

#include "sample_point.h"
#include <iostream>

SamplePoint::SamplePoint() {}

SamplePoint::SamplePoint(float x, float y, float z, float nx, float ny, float nz, float quality) : Point(x, y, z, nx, ny, nz) {
    this->quality=quality;
}

SamplePoint::SamplePoint(float x, float y, float z, float nx, float ny, float nz) : Point(x, y, z, nx, ny, nz) {}

SamplePoint::SamplePoint(float *pos, float *direction) : Point(pos, direction) {}

SamplePoint::SamplePoint(const SamplePoint &samplePoint): Point(samplePoint), quality(samplePoint.quality){
//    this->quality=samplePoint.quality;
}

float *SamplePoint::getPos() {
    return this->_pos;
}

float *SamplePoint::getDirection() {
    return this->_direction;
}

void SamplePoint::printInfo() {
    std::cout << "samplePoint info: address:" << &this->_pos << "pos[" << this->_pos[0] << ", " << this->_pos[1] << ", " << this->_pos[2]
              << "], address:" << &this->_direction << "direction[" << this->_direction[0] << ", " << this->_direction[1] << ", " << this->_direction[2] << "]" << std::endl;
}

SamplePoint &SamplePoint::operator=(SamplePoint &sp) {
    for(int i=0; i<3; i++){
        this->_pos[i] = sp.getPos()[i];
        this->_direction[i] = sp.getDirection()[i];
    }
    this->quality=sp.quality;
    return *this;
}
