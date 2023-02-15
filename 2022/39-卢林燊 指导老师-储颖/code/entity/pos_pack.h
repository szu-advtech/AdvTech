#ifndef VIEWCREATOR_POS_PACK_H
#define VIEWCREATOR_POS_PACK_H

#include <Eigen/Core>
#include <Eigen/Geometry>

class PosPack{
public:
    float yaw;
    float pitch;
    Eigen::Vector3f posMesh;
    Eigen::Vector3f direction;
    Eigen::Isometry3f cameraMatrix;
    PosPack()= default;
};

#endif //VIEWCREATOR_POS_PACK_H
