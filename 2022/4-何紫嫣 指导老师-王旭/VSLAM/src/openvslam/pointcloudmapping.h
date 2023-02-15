/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "openvslam/system.h"
#include "openvslam/Voxel.h"
#include "openvslam/CUDATSDFIntegrator.h"
#include <pcl/common/transforms.h>
#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <cstdlib>
#include <ctime>
#include <pcl/filters/statistical_outlier_removal.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <unordered_set>
using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// struct PointCloude
// {
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     PointCloud::Ptr pcE;
//     Eigen::Isometry3d T;
//     int pcID; 
//     int testID;
//     PointCloude(){}
// };

namespace openvslam {

namespace data {
class keyframe;
}

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    
    PointCloudMapping( cv::FileStorage tsdfSettings );
    void save();
    int3_label getint3(float x, float y, float z, double* pose);
    int3_label getbound(const Eigen::Vector3d &point);
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( data::keyframe* kf, cv::Mat& color, int idk, vector<data::keyframe*> vpKFs );
    // void insertKeyFrameM( KeyFrame* kf, cv::Mat& color, int idk,vector<KeyFrame*> vpKFs );
    void shutdown();
    void viewer();
    void generatePointCloud();
    // void inserttu( cv::Mat& color, cv::Mat& depth,int idk);
    // int loopcount = 0;
    vector<data::keyframe*> currentvpKFs;
    bool cloudbusy;
    bool loopbusy;
    // void updatecloud();
    bool bStop = false;
    // PotreeConverter* pc;
protected:
    // PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    
    PointCloud::Ptr globalMap;
    shared_ptr<thread>  viewerThread;
    shared_ptr<thread>  netThread;
    
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;  
    
    condition_variable  keyFrameUpdated;
    condition_variable  PointCloudeUpdated;
    mutex               keyFrameUpdateMutex;
    mutex               PointCloudeUpdateMutex;
    // vector<PointCloude, Eigen::aligned_allocator<PointCloude>>     pointcloud;
    // // data to generate point clouds
    // vector<KeyFrame*>       keyframes;
    // vector<cv::Mat>         colorImgs;
    queue<cv::Mat>          colorImgs;      // 待处理的图片队列
    queue<int>              colorID;        // 测试用
    // queue<double*>          poses;
    // queue<double*>          poses_inv;
    vector< Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> >     Ts;
    uint16_t                allNum = 0;     // 已传入的kf总数
    uint16_t                viewerSize =0;  // 已放到指定位置并加入globalMap的点云数量
    uint16_t                depthSize =0;   // 已估计深度构造点云的数量
    // vector<cv::Mat>         colorImgs;
    // vector<cv::Mat>         depthImgs;
    // vector<cv::Mat>         colorImgks;
    // vector<cv::Mat>         depthImgks;
    // vector<int>             ids;
    mutex                   keyframeMutex;
    // uint16_t                lastKeyframeSize =0;
    torch::jit::script::Module module;
    std::vector<torch::jit::IValue> input;
    
    // double resolution = 0.04;
    // double meank = 50;
    // double thresh = 1;
    // pcl::VoxelGrid<PointT>  voxel;
    // pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    CUDATSDFIntegrator Fusion;
    unordered_map<int3_label, VoxelBlock*> vmp;
    float tsdf_truncation;
    float tsdf_resolution;
    float tsdf_voxelsize;
    int kf_num;
};
}
#endif // POINTCLOUDMAPPING_H
