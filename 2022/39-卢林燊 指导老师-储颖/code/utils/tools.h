//
// Created by lj on 2022/11/6.
//

#ifndef FUXIAN_TOOLS_H
#define FUXIAN_TOOLS_H


#include <Eigen/Geometry>
#include "../entity/pos_pack.h"
#include "io_utils.h"

class Tools {
public:
    static Eigen::Isometry3f getCameraMatrix(const float yaw, const float pitch, const Eigen::Vector3f& vPos);
    static PosPack getPosPackFromDirectionVector(const float *vPos, const float *vDirection);
    static std::string makePath(const std::string &outputPath, const std::string &folderName, const std::string &fileName);
};


#endif //FUXIAN_TOOLS_H
