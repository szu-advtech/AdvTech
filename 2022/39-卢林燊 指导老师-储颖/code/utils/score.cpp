//
// Created by lj on 2022/11/8.
//

#include "score.h"
#include "vector"
#include "Params.h"

#define M_PI 3.1415926

// 构造函数
ScoreUtils::ScoreUtils(std::vector<int*> &faces, std::vector<SamplePoint> &meshPoint,
                       std::vector<SamplePoint> &samplePoints): points(samplePoints), partSamplePoints(samplePoints){

    // 初始化相机内参矩阵
    intrinsicMatrix << Params::RESOLUTION_H / 2 / std::tan(Params::FOV_H / 180 * M_PI / 2), 0,
            3000, 0, Params::RESOLUTION_V / 2 / std::tan(Params::FOV_V / 180 * M_PI / 2), 2000, 0, 0,
            1;

    for (int i = 0; i < meshPoint.size(); ++i) {
        Point_3 point(meshPoint[i].getPos()[0],
                      meshPoint[i].getPos()[1],
                      meshPoint[i].getPos()[2]);
        Point_3 normal(meshPoint[i].getDirection()[0],
                       meshPoint[i].getDirection()[1],
                       meshPoint[i].getDirection()[2]);
        vertices.push_back(point);
        normals.push_back(normal);
    }

    //三角面片的顶点集合
    for (int i=0;i<faces.size();i++){
        triangles.emplace_back(vertices[faces[i][0]],vertices[faces[i][1]],vertices[faces[i][2]]);
    }
    // 生成场景
    generateEmbreeScene(vertices, faces);
}

// 生成场景
void ScoreUtils::generateEmbreeScene(const std::vector<Point_3> &pVertices, const std::vector<int *> &faceIndexes) {
    RTCDevice   g_device = rtcNewDevice(NULL);
    RTCScene    scene    = rtcNewScene(g_device);
    RTCGeometry d_mesh   = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    struct Vertex
    {
        float x, y, z;
    };

    struct Triangle
    {
        int v0, v1, v2;
    };

    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
            d_mesh,
            RTC_BUFFER_TYPE_VERTEX,
            0,
            RTC_FORMAT_FLOAT3,
            sizeof(Vertex),
            pVertices.size()
    );

    Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
            d_mesh,
            RTC_BUFFER_TYPE_INDEX,
            0,
            RTC_FORMAT_UINT3,
            sizeof(Triangle),
            faceIndexes.size()
    );

    int i_index = 0;
    for(const auto& vertex:pVertices){
        vertices[i_index].x = vertex.x();
        vertices[i_index].y = vertex.y();
        vertices[i_index].z = vertex.z();
        i_index++;
    }
    i_index = 0;
    for (const auto& face : faceIndexes){
        triangles[i_index].v0 = face[0];
        triangles[i_index].v1 = face[1];
        triangles[i_index].v2 = face[2];
        i_index++;
    }

    rtcCommitGeometry(d_mesh);
    rtcAttachGeometry(scene, d_mesh);
    rtcReleaseGeometry(d_mesh);
    rtcCommitScene(scene);
    this->scene = scene;
}

// 计算视点可见性
bool ScoreUtils::isVisible(const Eigen::Matrix3f &vIntrinsicMatrix,
                           const Eigen::Isometry3f &vCameraMatrix,
                           const float *viewPos,
                           const float *pointPos,
                           const RTCScene &scene,
                           double maxDistance) {
    Eigen::Vector3f vViewPos(viewPos[0], viewPos[1], viewPos[2]);
    Eigen::Vector3f vPointPos(pointPos[0], pointPos[1], pointPos[2]);
    Eigen::Vector3f view_to_point = vPointPos - vViewPos;  // 从视点指向采样点的向量
    double view_to_origin_point = view_to_point.norm();  //向量的二范数，即欧式距离
    // 注意：normalized()与normalize()类似，只不过normalize()是对自身上做修改，而normalized()返回的是一个新的Vector/Matrix，并不改变原有的矩阵。
    view_to_point.normalize();  //把自身的各元素除以它的范数
    // 判断距离是否超出最大Distance
    if (maxDistance != -1.)
        if (maxDistance < view_to_origin_point) return false;
    //计算机采样点在相机成像的区域位于哪个像素位置
    Eigen::Vector3f pixel_pos = vIntrinsicMatrix * vCameraMatrix * vPointPos;
    if (pixel_pos.z() < 0) return false;  //小于零说明
    if (pixel_pos.x() < 0 || pixel_pos.x() > vIntrinsicMatrix(0, 2) * 2 ||
        pixel_pos.y() < 0 || pixel_pos.y() > vIntrinsicMatrix(1, 2) * 2)
        return false;

    // 计算是否相交
    RTCRayHit rayhits;
    rayhits.ray.org_x  = static_cast<float>(vViewPos.x());
    rayhits.ray.org_y  = static_cast<float>(vViewPos.y());
    rayhits.ray.org_z  = static_cast<float>(vViewPos.z());
    rayhits.ray.dir_x  = static_cast<float>(view_to_point.x());
    rayhits.ray.dir_y  = static_cast<float>(view_to_point.y());
    rayhits.ray.dir_z  = static_cast<float>(view_to_point.z());
    rayhits.ray.tnear  = 0;
    rayhits.ray.tfar   = std::numeric_limits<float>::infinity();
    rayhits.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhits.hit.primID = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    rtcIntersect1(scene, &context, &rayhits);

    if (rayhits.ray.tfar < 0) return false;
    if (rayhits.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        auto hit_position = vViewPos + (view_to_point * rayhits.ray.tfar);
        return rayhits.ray.tfar >= view_to_origin_point
               || (hit_position - vPointPos).norm() <= 1e-3;
    } else {
        return false;
    }
}

void ScoreUtils::computeVisibility() {
    // 重置可见性点集合的大小
    pointViewVisibilitySet.resize(points.size(), std::vector<int>());
    viewPointVisibilitySet.resize(this->trajectory.size(), std::vector<int>());

    for (int i_point = 0; i_point < points.size(); ++i_point) {
        for (int j = 0; j < trajectory.size(); j++){
            if (isVisible(intrinsicMatrix,
                          this->trajectory[j].cameraMatrix,
                          this->trajectory[j].getPos(),
                          points[i_point].getPos(),
                          scene,
                          Params::MAX_D))
            {
                pointViewVisibilitySet[i_point].push_back(j);
                // size问题
                viewPointVisibilitySet[j].push_back(i_point);
            }

        }
    }
}

void ScoreUtils::setTrajectory(std::vector<ViewPoint> &trajectory) {
// 清空相关变量
    this->pointViewVisibilitySet.clear();
    this->viewPointVisibilitySet.clear();
    this->trajectory = trajectory;
}

void ScoreUtils::updateVisibility(){

    this->computeVisibility();
}