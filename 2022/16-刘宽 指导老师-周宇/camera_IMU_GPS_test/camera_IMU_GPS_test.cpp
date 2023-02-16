/*
说明：
本程序尝试：相机+IMU+GPS， 其实VINS-fusion 已经具备这个条件了，主要在于结合；
本次程序是rosNodeTest.cpp 和 KITTIGPS.cpp 的结合；
通过读取不同的配置文件，可以决定是单目相机还是双目相机；
先尝试用Kitti的数据，在尝试用自己采集的数据！
*/
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <map>
#include <thread>
#include <mutex>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include "estimator/estimator.h"
#include "utility/visualization.h"
#include "camera_imu_gps.h"
 
 
using namespace std;
using namespace Eigen;
 
Estimator estimator;
ros::Publisher pubGPS;
 
std::mutex m_buf;
 
 
void readIMUdata(const std::string &line, IMU_MSG &imuObs);
void readGPSdata(const std::string &line, GPS_MSG &gpsObs);
 
 
 
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}
 
 
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}
 
void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}
 
void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}
 
 
 
int main(int argc, char** argv)
{
  	ros::init(argc, argv, "vins_estimator");
	ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
 
	pubGPS = n.advertise<sensor_msgs::NavSatFix>("/gps", 1000);
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);
 
    
	if(argc != 3)
	{
		printf("please intput: rosrun vins camera_IMU_GPS_test [config file] [data folder] \n"
			   "for example: rosrun vins camera_IMU_GPS_test "
			   "~/catkin_ws/src/VINS-Fusion/config/test_data/stereo_imu_gps_config.yaml "
			   "/home/hltt3838/kitti_data/2011_10_03_drive_0042_sync/ \n");
		return 1;
	}
    
	string config_file = argv[1];//stereo_imu_gps_config.yaml
	printf("config_file: %s\n", argv[1]);
    
	string sequence = argv[2];   //---/home/hltt3838/kitti_data/2011_10_03_drive_0042_sync/
	printf("read sequence: %s\n", argv[2]);
	string dataPath = sequence + "/";  
    
    //1、读取imu的 txt 文件，一行一行读取
    double init_imu_time;
    IMU_MSG imuObs;
    std::string line_imu;
    std::string imuPath = dataPath + "imu_data_100hz/imu.txt";
    std::ifstream  csv_IMUfile(imuPath); 
    if (!csv_IMUfile)
    {
	    printf("cannot find imu Path \n" );
	    return 0;          
	}
    std::getline(csv_IMUfile, line_imu); //header, 获得的第一个IMU数据
    
    readIMUdata(line_imu, imuObs);
    init_imu_time = imuObs.time;
    printf("init_imu_time： %10.5f \n", init_imu_time);
	
  
    //2、读取GPS的 txt 文件，一行一行读取
    double init_gps_time;
    GPS_MSG gpsObs;
    std::string line_gps;
    std::string gpsPath = dataPath + "gps_data_10hz/gps.txt";
    std::ifstream  csv_GPSfile(gpsPath);
    if (!csv_GPSfile)
    {
	    printf("cannot find gps Path \n" );
	    return 0;          
	}
    std::getline(csv_GPSfile, line_gps); //header, 获得的第一个gps数据
    readGPSdata(line_gps, gpsObs);
    init_gps_time = gpsObs.time;
    printf("init_gps_time： %10.5f \n", init_gps_time); 
    
    
    //3、读取图像时间，整个文件读取，存放到 imageTimeList,两个相机对齐了，没有进行判断
    //Cam0
    double init_cam_time;
	FILE* file;
	file = std::fopen((dataPath + "image_00/timestamps.txt").c_str() , "r");
	if(file == NULL)
    {
	    printf("cannot find file: %simage_00/timestamps.txt \n", dataPath.c_str());
	    ROS_BREAK();
	    return 0;          
	}
	vector<double> image0TimeList;
	int year, month, day;
	int hour, minute;
	double second;
	while (fscanf(file, "%d-%d-%d %d:%d:%lf", &year, &month, &day, &hour, &minute, &second) != EOF)
	{
	    image0TimeList.push_back(hour * 60 * 60 + minute * 60 + second);
	}
	std::fclose(file);
    
    
    init_cam_time = image0TimeList[0]; 
    printf("init_cam_time: %10.5f \n", init_cam_time);
    
    double baseTime;
    baseTime = min(init_imu_time,init_gps_time,init_cam_time);
    printf("baseTime: %10.5f \n", baseTime);
    
    //4、读取配置参数和发布主题
    readParameters(config_file);
	estimator.setParameter();
	registerPub(n);
    
    //5、VIO的结果输出保存文件
    FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(),"w");
	if(outFile == NULL)
    printf("Output vio path dosen't exist: %s\n", OUTPUT_FOLDER.c_str());
	string leftImagePath, rightImagePath;
	cv::Mat imLeft, imRight;
    
    
    //6、遍历整个图像
    for (size_t i = 0; i < image0TimeList.size(); i++)
	{	
        int num_imu = 0;
		if(ros::ok())
		{
			printf("process image %d\n", (int)i);
			stringstream ss;
			ss << setfill('0') << setw(10) << i;
			leftImagePath = dataPath + "image_00/data/" + ss.str() + ".png";
			rightImagePath = dataPath + "image_01/data/" + ss.str() + ".png";
            
			printf("%s\n", leftImagePath.c_str() );
			printf("%s\n", rightImagePath.c_str() );
            double imgTime = 0; 
           
            imLeft  = cv::imread(leftImagePath, CV_LOAD_IMAGE_GRAYSCALE );
            imRight = cv::imread(rightImagePath, CV_LOAD_IMAGE_GRAYSCALE );
            
            imgTime = image0TimeList[i] - baseTime;
            printf("image time: %10.5f \n", imgTime);
           
            
			//读取GPS信息
            std::getline(csv_GPSfile, line_gps);  
            readGPSdata(line_gps, gpsObs);
            
            sensor_msgs::NavSatFix gps_position;
			gps_position.header.frame_id = "NED";
			gps_position.header.stamp = ros::Time(imgTime);
			gps_position.latitude  = gpsObs.position[0];
			gps_position.longitude = gpsObs.position[1];
			gps_position.altitude  = gpsObs.position[2];
			gps_position.position_covariance[0] = gpsObs.pos_accuracy;
    
			pubGPS.publish(gps_position);
            
   //---------------------加上IMU-------------------------//       
            //读取imu的信息
            while (std::getline(csv_IMUfile, line_imu))
            {
               num_imu++;
               printf("process imu %d\n",num_imu);
               readIMUdata(line_imu, imuObs);
               double imuTime = imuObs.time - baseTime;
               printf("imu time: %10.5f \n", imuTime);
                
               Vector3d acc = imuObs.acc;
               Vector3d gyr = imuObs.gyr;
                   
               estimator.inputIMU(imuTime, acc, gyr);
                   
               if (imuTime >= imgTime) //简单的时间同步,IMU的时间戳是不大于图像的
                {
                    break;
                }
            }
     //---------------------加上IMU-------------------------//     
     
            if(STEREO)//双目为1，否则为0
            {
              estimator.inputImage(imgTime,imLeft, imRight);
            }
            else
              estimator.inputImage(imgTime, imLeft);
 
        
			Eigen::Matrix<double, 4, 4> pose;
			estimator.getPoseInWorldFrame(pose);
			if(outFile != NULL)
				fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
																	       pose(1,0), pose(1,1), pose(1,2),pose(1,3),
																	       pose(2,0), pose(2,1), pose(2,2),pose(2,3));
			
			 //cv::imshow("leftImage", imLeft);
			 //cv::imshow("rightImage", imRight);
			 //cv::waitKey(2);
		}
		else
			break;
	}
	
	if(outFile != NULL)
		fclose (outFile);
    
    
    return 0;
}   
 
 
 
 
 
 
void readIMUdata(const std::string &line, IMU_MSG &imuObs)
{
    std::stringstream  lineStream(line);
    std::string        dataRecord[7];
    std::getline(lineStream, dataRecord[0], ' ');//这里的数据间是空格， 如果有逗号，用'，'
    std::getline(lineStream, dataRecord[1], ' ');
    std::getline(lineStream, dataRecord[2], ' ');
    std::getline(lineStream, dataRecord[3], ' ');
    std::getline(lineStream, dataRecord[4], ' ');
    std::getline(lineStream, dataRecord[5], ' ');
    std::getline(lineStream, dataRecord[6], ' ');
    
    imuObs.time = std::stod(dataRecord[0]); //时间：s;
             
    imuObs.acc[0] = std::stod(dataRecord[1]);
    imuObs.acc[1] = std::stod(dataRecord[2]);
    imuObs.acc[2] = std::stod(dataRecord[3]);
 
    imuObs.gyr[0] = std::stod(dataRecord[4]);
    imuObs.gyr[1] = std::stod(dataRecord[5]);
    imuObs.gyr[2] = std::stod(dataRecord[6]);
}
 
 
void readGPSdata(const std::string &line, GPS_MSG &gpsObs)
{
    std::stringstream  lineStream(line);
    std::string        dataRecord[7];
    std::getline(lineStream, dataRecord[0], ' ');//这里的数据间是空格， 如果有逗号，用'，'
    std::getline(lineStream, dataRecord[1], ' ');
    std::getline(lineStream, dataRecord[2], ' ');
    std::getline(lineStream, dataRecord[3], ' ');
    std::getline(lineStream, dataRecord[4], ' ');
 
    gpsObs.time = std::stod(dataRecord[0]); //时间：s;
             
    gpsObs.position[0] = std::stod(dataRecord[1]);
    gpsObs.position[1] = std::stod(dataRecord[2]);
    gpsObs.position[2] = std::stod(dataRecord[3]);
 
    gpsObs.pos_accuracy = std::stod(dataRecord[4]);
   
}