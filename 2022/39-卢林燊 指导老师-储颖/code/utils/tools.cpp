//
// Created by lj on 2022/11/6.
//

#include "tools.h"

#define M_PI 3.1415926

Eigen::Isometry3f Tools::getCameraMatrix(const float yaw, const float pitch, const Eigen::Vector3f &vPos) {
    Eigen::Vector3f posMesh(vPos[0], vPos[1], vPos[2]);
    Eigen::Isometry3f result = Eigen::Isometry3f::Identity();
    result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())); // 偏航角
    result.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())); // 俯仰角
    result.rotate(Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitX()));
    result.rotate(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitY()));
    result.pretranslate(posMesh);
    return result.inverse();  // 返回矩阵的逆
}

PosPack Tools::getPosPackFromDirectionVector(const float *vPos, const float *vDirection){
    Eigen::Vector3f pos(vPos[0], vPos[1], vPos[2]);
    Eigen::Vector3f direction(vDirection[0], vDirection[1], vDirection[2]);
    PosPack pos_pack;
    Eigen::Vector3f normalized_direction = direction.normalized();
    pos_pack.yaw = std::atan2(normalized_direction.y(), normalized_direction.x());
    pos_pack.pitch = -std::asin(normalized_direction.z());
    pos_pack.posMesh = pos;
    //pos_pack.pos_airsim = convertMeshToUnreal(convertUnrealToAirsim(v_pos_mesh));
    pos_pack.cameraMatrix = getCameraMatrix(pos_pack.yaw, pos_pack.pitch, pos_pack.posMesh);
    pos_pack.direction = Eigen::Vector3f(
            std::cos(pos_pack.pitch)*std::cos(pos_pack.yaw),
            std::cos(pos_pack.pitch)*std::sin(pos_pack.yaw),
            std::sin(-pos_pack.pitch)).normalized();
    return pos_pack;
}

std::string Tools::makePath(const std::string &outputPath, const std::string &folderName, const std::string &fileName) {
    return outputPath + "\\" + folderName + "\\" + fileName;
}