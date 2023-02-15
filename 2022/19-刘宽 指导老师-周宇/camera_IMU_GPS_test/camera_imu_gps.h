#pragma once
#include<chrono>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
 
struct IMU_MSG
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
    IMU_MSG &operator =(const IMU_MSG &other)
    {
        time = other.time;
        acc = other.acc;
        gyr = other.gyr;
        return *this;
    }
};
 
struct GPS_MSG
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time;
    Eigen::Vector3d position;
    double pos_accuracy;
    GPS_MSG &operator =(const GPS_MSG &other)
    {
        time         = other.time;
        position     = other.position;
        pos_accuracy = other.pos_accuracy;
        return *this;
    }
};
 
double min(double x,double y,double z )
{
    return x < y ? (x < z ? x : z) : (y < z ? y : z);
}
