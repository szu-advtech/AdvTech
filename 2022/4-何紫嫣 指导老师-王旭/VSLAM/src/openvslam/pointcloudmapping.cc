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
#include "openvslam/type.h"
#include "openvslam/pointcloudmapping.h"
#include <openvslam/data/keyframe.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry> 
#include "openvslam/util/converter.h"
// #include "openvslam/PointCloude.h"
#include "openvslam/system.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
using namespace std;
float m_PI = 3.14159265358979323846f;
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int currentloopcount = 0;
double timepc = 0;
double timetsdf = 0;
auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCPU)
    .requires_grad(false);
extern "C" void testloading(Voxel* d_SDFBlocks);

void printpose(double *pose)
{
    for (int i=0;i<16;i++) pose[i] = pose[i];
}

namespace openvslam {

PointCloudMapping::PointCloudMapping(cv::FileStorage tsdfSettings)
{
    
    // this->resolution = resolution_;
    // this->meank = thresh_;
    // this->thresh = thresh_;
    module = torch::jit::load("/home/code/pcldemo/traced_model.pt");
    cout << torch::cuda::is_available() << endl;
    Fusion.Initialize(tsdfSettings);
    tsdf_truncation = tsdfSettings["Truncation"];
    tsdf_resolution = tsdfSettings["Resolution"];
    tsdf_voxelsize = tsdfSettings["VoxelSize"];
    kf_num = tsdfSettings["KFnum"];
    // statistical_filter.setMeanK(meank);
    // statistical_filter.setStddevMulThresh(thresh);
    // voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    loopbusy = false;
    cloudbusy = false;
    // Voxel* vx = new Voxel;
    // vx->weight = 5.5;
    // Voxel *vc;
    // cudaMalloc(&vc, sizeof(Voxel));
    // cudaMemcpy(vc, vx, sizeof(Voxel), cudaMemcpyHostToDevice);
    // testloading(vc);
    // cudaFree(vc);
    // delete vx;
    // pc = new PotreeConverter("./potree_converted");
    netThread = make_shared<thread>( bind(&PointCloudMapping::generatePointCloud, this ) );
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

int3_label PointCloudMapping::getint3(float x, float y, float z, double* pose)
{
    // float xx = x + pose[0 * 4 + 3];
    // float yy = y + pose[1 * 4 + 3];
    // float zz = z + pose[2 * 4 + 3];
    x = pose[0 * 4 + 0] * x + pose[1 * 4 + 0] * y + pose[2 * 4 + 0] * z + pose[3 * 4 + 0];
    y = pose[0 * 4 + 1] * x + pose[1 * 4 + 1] * y + pose[2 * 4 + 1] * z + pose[3 * 4 + 1];
    z = pose[0 * 4 + 2] * x + pose[1 * 4 + 2] * y + pose[2 * 4 + 2] * z + pose[3 * 4 + 2];
    int3_label lab;
    lab.x = floor(x/0.1);
    lab.y = floor(y/0.1);
    lab.z = floor(z/0.1);
    return lab;
}

int3_label PointCloudMapping::getbound(const Eigen::Vector3d &point)
{
    int3_label lab;
    lab.x = (int)floor(point(0)/(tsdf_resolution*tsdf_voxelsize));
    lab.y = (int)floor(point(1)/(tsdf_resolution*tsdf_voxelsize));
    lab.z = (int)floor(point(2)/(tsdf_resolution*tsdf_voxelsize));
    return lab;
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
        PointCloudeUpdated.notify_one();
    }
    netThread->join();
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(data::keyframe* kf, cv::Mat& color, int idk,vector<data::keyframe*> vpKFs)
{
    // ofstream outfile("./pose.txt",ios::app);
    cout<<"insertKeyFrame, img size is [ " << color.rows << " , " << color.cols << " ] " <<endl;
    unique_lock<mutex> lck(keyframeMutex);
    allNum++;
    currentvpKFs = vpKFs;
    colorImgs.push(color.clone());
    colorID.push(allNum);
    Eigen::Isometry3d T = util::converter::to_g2o_SE3( kf->get_cam_pose() );
    Ts.push_back(T);
    // double* pose = new double[16];
    // Eigen::Map<Eigen::MatrixXd>(pose, T.matrix().rows(), T.matrix().cols()) = T.matrix().transpose();
    // cout << "Eigen " << allNum << " : " << endl << util::converter::to_g2o_SE3( kf->get_cam_pose() ) << endl;
    // cout << "pose array: " << endl;
    // for (int i=0;i<4;i++)
    // {
    //     for (int j=0;j<4;j++)
    //         cout << pose[i*4+j] << " ";
    //     cout << endl;
    // }
    // double* pose_inv = new double[16];
    // Eigen::Map<Eigen::MatrixXd>(pose_inv, T.inverse().matrix().rows(), T.inverse().matrix().cols()) = T.inverse().matrix().transpose();
    // poses.push(pose);
    // poses_inv.push(pose_inv);


    // PointCloude pointcloude;
    // pointcloude.pcID = idk;
    // pointcloude.T = util::converter::to_g2o_SE3( kf->get_cam_pose() );
    // cout << "????" << pointcloude.T.matrix() * pointcloude.T.inverse().matrix() << endl;
    // pointcloud.push_back(pointcloude);
    keyFrameUpdated.notify_one();
    // Eigen::Vector3d p(0.132593,-0.397363,-0.269568); // 0.13256 -0.415168 -0.281343
    // cout << (pointcloude.T * pointcloude.T.inverse() * p).transpose() << endl;
    
    // Eigen::Quaterniond qq(pointcloude.T.rotation());
    // outfile << qq.coeffs().transpose() << " " << pointcloude.T.translation().transpose() << endl;
    // outfile.close();
}


void PointCloudMapping::generatePointCloud()
{
    cv::Mat color_im;
    int testID;
    // PointCloude *pc;
    double *pose;
    double *pose_inv;
    int im_width = 1920;
    int im_height = 960;
    uint8_t color[im_width * im_height * 3];
    float depth[1024*512];
    Eigen::Isometry3d T;
    while (1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        cout << "generatePointCloud" << endl;
        input.clear();
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = allNum;
        }
        while(1)
        {
            // 取待添加数据的 pc 和对应的 color
            {
                unique_lock<mutex> lck(keyframeMutex);
                if (depthSize>=allNum)break;
                color_im = colorImgs.front();
                colorImgs.pop();
                T = Ts[depthSize];
                pose = T.matrix().data();
                cout << "pose 地址:" << pose << endl;
                
                testID = colorID.front();
                colorID.pop();
                // cout << "Eigen " << depthSize << " : " << endl << T.matrix() << endl;
            }
            cout << "TODO: get pc " << testID << endl;
            cout << "0. 预测深度图，准备数据 ";
            cv::Mat img;
            cv::resize(color_im, img, cvSize(1024, 512));
            img.convertTo(img, CV_32F, 1/255.0);
            auto input_tensor = torch::from_blob(img.data, {1, 512, 1024, 3}, options).clone();
            input_tensor = input_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA);
	        input.emplace_back(input_tensor);
            auto model_output = module.forward(input).toTensor();
            auto depth_im = model_output.squeeze().to(torch::kF32).to(torch::kCPU);
            input.clear();
            cout << "get depth" << endl;

            memcpy(color, color_im.data, sizeof(uint8_t) * 3 * im_height * im_width); 
            for(int r = 0; r < 512; r++)
                for(int c = 0; c < 1024; c++)
                {
                    depth[r * 1024 + c] = depth_im[r][c].item().toFloat();
                    if(depth[r * 1024 + c] > 6.0)
                        depth[r * 1024 + c] = 0.0; // Only consider depth < 6m
                }

            unordered_set<int3_label> vst;
            vst.clear();
            cout << "1.生成点云并转换至世界坐标系";
            PointCloud::Ptr pt (new PointCloud);
            PointCloud::Ptr pt_trans (new PointCloud);
            clock_t start1 = clock();
            float min_x = 0.0f, min_y = 0.0f, min_z = 0.0f;
            float max_x = 0.0f, max_y = 0.0f, max_z = 0.0f;

            for ( int v=0; v<color_im.rows; v++ )
                for ( int u=0; u<color_im.cols; u++ )
                {
                    int vv = (int)(v*1.0/1.875);
                    int uu = (int)(u*1.0/1.875);
                    float td = depth_im[vv][uu].item().toFloat();
                    // float td = depth_im[v][u].item().toFloat();
                    if (td<=0 || td>5)continue;
                    if (v>900 && (u<100 || u>1900))continue; // 暂时挖掉，后面需要用周围颜色填补
                    td/=2.0;
                    float m = (u+0.5) / color_im.cols * 2 * m_PI;
                    float n = ((v+0.5) / color_im.rows-0.5) * m_PI;
                    float y = td * sin(n);
                    if (y>0.77 || y <= -0.6) continue;
                    float z = - td * cos(n) * cos(m);
                    float x = - td * cos(n) * sin(m);
                    if(sqrt(z*z+x*x)<0.3)continue;
                    PointT pp;
                    pp.x = x, pp.y = y, pp.z = z;
                    pp.b = color_im.data[ v*color_im.step+u*color_im.channels() ];
                    pp.g = color_im.data[ v*color_im.step+u*color_im.channels()+1 ];
                    pp.r = color_im.data[ v*color_im.step+u*color_im.channels()+2 ];
                    pt->points.push_back(pp);
                }
            pcl::transformPointCloud( *pt, *pt_trans, T.inverse().matrix()); // pt_trans中此时为该帧对应的世界坐标系下的点云
            *globalMap += *pt_trans;
            timepc += (double)(clock()-start1);
            cout << "该帧有效点云点数目：" << pt_trans->points.size() << endl;
            cout << "2. 将世界坐标系下点云的bounding box中的每个小体素块放入vst" << endl;
            clock_t start2 = clock();
            for(int nIndex = 0; nIndex < pt_trans->points.size (); nIndex++)
            {
                min_x = (min_x > pt_trans->points[nIndex].x)? pt_trans->points[nIndex].x : min_x;
                min_y = (min_y > pt_trans->points[nIndex].y)? pt_trans->points[nIndex].y : min_y;
                min_z = (min_z > pt_trans->points[nIndex].z)? pt_trans->points[nIndex].z : min_z;
                max_x = (max_x < pt_trans->points[nIndex].x)? pt_trans->points[nIndex].x : max_x;
                max_y = (max_y < pt_trans->points[nIndex].y)? pt_trans->points[nIndex].y : max_y;
                max_z = (max_z < pt_trans->points[nIndex].z)? pt_trans->points[nIndex].z : max_z;
            }
            int3_label min_bound = getbound(Eigen::Vector3d(min_x, min_y, min_z));
            int3_label max_bound = getbound(Eigen::Vector3d(max_x, max_y, max_z));
            cout << "转换后世界坐标下min bound： " << min_bound.x << " " << min_bound.y << " " << min_bound.z << endl;
            cout << "转换后世界坐标下max bound： " << max_bound.x << " " << max_bound.y << " " << max_bound.z << endl;
            for (int i = min_bound.x; i<=max_bound.x; i++)
                for (int j = min_bound.y; j<=max_bound.y; j++)
                    for (int k = min_bound.z; k<=max_bound.z; k++)
                    {
                        int3_label lab;
                        lab.x = i, lab.y = j, lab.z = k;
                        auto pr = vst.insert(lab);
                        if(pr.second) 
                            vmp.insert(make_pair(lab, nullptr));
                    }
            cout << "用时 " << (double)(clock()-start2)/CLOCKS_PER_SEC << " s" << endl;
            cout << "该帧包含体素块个数：" << vst.size() << endl;

            cout << "3. 遍历set，将label和VoxelBlock复制到GPU上" << endl;
            clock_t start3 = clock();
            Fusion.uploadData(depth, color, pose, vst, vmp);
            cout << "用时 " << (double)(clock()-start3)/CLOCKS_PER_SEC << " s" << endl;

            cout << "4. GPU更新该帧包含的所有体素块" << endl;
            clock_t start4 = clock();
            Fusion.integrate();
            timetsdf += (double)(clock()-start4);
            cout << "用时 " << (double)(clock()-start4)/CLOCKS_PER_SEC << " s" << endl;

            cout << "5. 遍历set，将label和VoxelBlock从GPU上取出放回map" << endl;
            clock_t start5 = clock();
            Fusion.downloadData(vst, vmp);
            cout << "用时 " << (double)(clock()-start5)/CLOCKS_PER_SEC << " s" << endl;

            if(testID>=kf_num)
            {
                save();
                break;
            }

            // 放入生成好的数据
            {
                unique_lock<mutex> lck(keyframeMutex);
                depthSize++;
                // PointCloudeUpdated.notify_one();
                if (depthSize>=allNum)break;
            }
        }
    }
}


void PointCloudMapping::viewer()
{
    // pcl::visualization::CloudViewer viewer("viewer");
    cout << "new a viewer!" << endl;
    int cnt = 0;
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_pointcloudeUpdated( PointCloudeUpdateMutex );
            PointCloudeUpdated.wait( lck_pointcloudeUpdated );
        }
        cout << "viewing" << endl;
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = depthSize;
        }
        if(loopbusy || bStop)
        {
          cout<<"loopbusy || bStop"<<endl;
            continue;
        }
        cout<<viewerSize<<"    "<<N<<endl;
        if(viewerSize == N)
        {
            cloudbusy = false;
            continue;
        }
            
        cloudbusy = true;
        for ( size_t i=viewerSize; i<N ; i++ )
        {
            // PointCloud::Ptr p (new PointCloud);
            // cout << "当前总插入点云数量" << pointcloud.size() << endl;
            // pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());
            // cout<<"处理好第" << i << "个点云， (pcID - testID) ： (" << pointcloud[i].pcID << " - " << pointcloud[i].testID << " )" <<endl;
            // *globalMap += *p;
            // PointCloud::Ptr tmp1(new PointCloud);
            // statistical_filter.setInputCloud(globalMap);
            // statistical_filter.filter( *tmp1 );
            // string plyName = "./pctest/"+to_string(i)+".ply";
            // pcl::io::savePLYFileBinary( plyName, *tmp1 );
            // pcl::io::savePLYFileBinary( plyName, *globalMap );
            // cerr << "已保存" << endl;
        }
        // viewer.showCloud( globalMap );
        // cout<<"show global map, size="<<N<<"   "<<globalMap->points.size()<<endl;
        // if (viewerSize>50)
        // {
        //     save();
        //     shutdown();
        //     break;
        // }
        // cout<<"show global map"<<endl;
        viewerSize = N;
        cloudbusy = false;
    }
}
void PointCloudMapping::save()
{
    cout << "timepc(per) = " << timepc / (CLOCKS_PER_SEC*20) << endl;
    cout << "timetsdf(per) = " << timetsdf / (CLOCKS_PER_SEC*20) << endl;
    cout << "[save] 遍历map，生成全局点云" << endl;
    clock_t start5 = clock();
    Fusion.SaveVoxelGrid2SurfacePointCloud(vmp);
    cout << "用时 " << (double)(clock()-start5)/CLOCKS_PER_SEC << " s" << endl;
    // exit(1);
	// pcl::io::savePLYFileBinary( "result.ply", *globalMap );
    // pcl::io::savePCDFileBinary( "result.pcd", *globalMap );
	// cout<<"globalMap save finished"<<endl;

    // Create header for .ply file
    FILE *fp = fopen("result_test1.ply", "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", globalMap->points.size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");
    for(int i = 0; i < globalMap->points.size (); i++)
    {
        float x = globalMap->points[i].x;
        float y = globalMap->points[i].z;
        float z = -globalMap->points[i].y;
        fwrite(&x, sizeof(float), 1, fp);
        fwrite(&y, sizeof(float), 1, fp);
        fwrite(&z, sizeof(float), 1, fp);
        fwrite(&globalMap->points[i].r, sizeof(uchar), 1, fp);
        fwrite(&globalMap->points[i].g, sizeof(uchar), 1, fp);
        fwrite(&globalMap->points[i].b, sizeof(uchar), 1, fp);
    }
    fclose(fp);
    cout<<"globalMap save finished"<<endl;
}
// void PointCloudMapping::updatecloud()
// {
// 	if(!cloudbusy)
// 	{
// 		loopbusy = true;
// 		cout<<"startloopmappoint"<<endl;
//         PointCloud::Ptr tmp1(new PointCloud);
// 		for (int i=0;i<currentvpKFs.size();i++)
// 		{
// 		    for (int j=0;j<pointcloud.size();j++) 
// 		    {   
// 				if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId) 
// 				{   
// 					Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );
// 					PointCloud::Ptr cloud(new PointCloud);
// 					pcl::transformPointCloud( *pointcloud[j].pcE, *cloud, T.inverse().matrix());
// 					*tmp1 +=*cloud;

// 					cout<<"第pointcloud"<<j<<"与第vpKFs"<<i<<"匹配"<<endl;
// 					continue;
// 				}
// 			}
// 		}
//         cout<<"finishloopmap"<<endl;
//         PointCloud::Ptr tmp2(new PointCloud());
//         voxel.setInputCloud( tmp1 );
//         voxel.filter( *tmp2 );
//         globalMap->swap( *tmp2 );
//         //viewer.showCloud( globalMap );
//         loopbusy = false;
//         //cloudbusy = true;
//         loopcount++;

//         //*globalMap = *tmp1;
// 	}
// }

}