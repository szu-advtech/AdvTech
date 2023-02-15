#ifndef POINTCLOUDE_H
#define POINTCLOUDE_H

#include "openvslam/pointcloudmapping.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/core/core.hpp>
#include <mutex>

namespace openvslam
{

class PointCloude
{
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
public:
    //PointCloud::Ptr pcE;
    // PointCloude(Eigen::Isometry3d T_e, PointCloud::Ptr p, int idk, int tid):T(T_e),pcE(p),pcID(idk),testID(tid){
    //     // T = T_e;
    //     // pcE = p;
    //     // testID = -1;
    //     // pcID = idk;
    // }
    PointCloude(){}
public:
    Eigen::Isometry3d T;
    int pcID; 
    int testID; // 测试用
};

}

#endif // POINTCLOUDE_H
