//
// Created by lj on 2022/11/7.
//

#ifndef FUXIAN_PARAMS_H
#define FUXIAN_PARAMS_H
#define M_PI 3.1415926

namespace Params{
    const std::string SAMPLE_FILE_PATH = R"(D:\FuXian\data\xuexiao_fine_90.ply)";
    const std::string MESH_FILE_PATH = R"(D:\FuXian\data\xuexiao.ply)";

    const float MIN_DISTANCE_BETWEEN_POINT_AND_VIEW = 25;
    const float MAX_DISTANCE_BETWEEN_POINT_AND_VIEW = 35;
    const float PER_RADIAN = 360 / 36.0 / 180.0 * M_PI;
    const int POP_SIZE = 100;
    const int TOTAL_VIEW_NUMS = 559;
    const float MAX_D = 120;  //相机能观察的最远距离
    const float BEST_DISTANCE = 30;

    // 相机内参
    const double RESOLUTION_H = 6000;
    const double RESOLUTION_V = 4000;
    const double FOV_H = 60;
    const double FOV_V = 42;
}



#endif //FUXIAN_PARAMS_H
