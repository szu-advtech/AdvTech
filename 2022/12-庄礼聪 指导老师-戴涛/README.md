# ORBSLAM2_with_IMU
This is a modified ORB_SLAM2 (from https://github.com/raulmur/ORB_SLAM2)

# Prerequisites
We have tested the library in **Ubuntu 12.04**, **14.04** and **16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Sophus

# Building

Download the vovabulary at https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary and put in /Vocabulary/

Clone the repository:
```
cd ORB_SLAM2_with_IMU
chmod +x build.sh
./build.sh
```

#Run examples
Make sure the path is correct

```
cd ORB_SLAM2_with_IMU
./mh01_exmaple.sh
```



