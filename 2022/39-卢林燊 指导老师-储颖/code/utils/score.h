//
// Created by lj on 2022/11/8.
//

#ifndef FUXIAN_SCORE_H
#define FUXIAN_SCORE_H

#include <vector>
#include <Eigen/Geometry>
#include <embree3/rtcore.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include "../entity/sample_point.h"
#include "../entity/view_point.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Triangle_3 Triangle_3;

class ScoreUtils {
public:
    bool partFlag = false;

    Eigen::Matrix3f intrinsicMatrix; //相机内参矩阵

    RTCScene scene;  // 光线碰撞场景
    std::vector<Point_3> vertices;  //顶点坐标
    std::vector<Point_3> normals;   //法向量
    std::vector<Triangle_3> triangles;  //三角面片

    // 存储的是离散的采样点：使用引用
    std::vector<SamplePoint> &points;  //样本点
    std::vector<ViewPoint> trajectory;  //视点集合

    // 存储可见性结果：使用move复制
    std::vector<std::vector<int>> pointViewVisibilitySet;  //采样点的可见性集合,即存的是能看见采样点的所有视点，采样点1能被{视点1，视点2,....}看到
    std::vector<std::vector<int>> viewPointVisibilitySet;  //视点的可见性集合，即存的是能看见视点的所有采样点，视点1能看到{采样点1，采样点2,....}

    std::vector<SamplePoint> partSamplePoints;
    // 构造函数
    ScoreUtils(std::vector<int*> &faces, std::vector<SamplePoint> &meshPoint, std::vector<SamplePoint> &points);
    // 生成场景函数
    void generateEmbreeScene(const std::vector<Point_3> &pVertices,const std::vector<int*> &faceIndexes);

    //可见性判断函数
    bool isVisible(const Eigen::Matrix3f &vIntrinsicMatrix,
                   const Eigen::Isometry3f &vCameraMatrix,
                   const float *pos,
                   const float *pointPos,
                   const RTCScene &scene,
                   double maxDistance);


    void computeVisibility();
    void updateVisibility();
    void setTrajectory(std::vector<ViewPoint> &trajectory);


};


#endif //FUXIAN_SCORE_H
