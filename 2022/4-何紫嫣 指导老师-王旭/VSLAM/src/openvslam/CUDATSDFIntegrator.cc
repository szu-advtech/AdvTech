//
// Created by will on 20-1-9.
//

#include "openvslam/CUDATSDFIntegrator.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

extern "C" void IntegrateDepthMapCUDA(int3_label* d_labels, double* d_pose, float* d_depth, uint8_t* d_color, 
                                      float truncation, float resolution, float voxelSize, int height, int width, int grid_dim_x, int grid_dim_y, int grid_dim_z, Voxel* d_SDFBlocks, int num);


CUDATSDFIntegrator::CUDATSDFIntegrator(){}

void CUDATSDFIntegrator::Initialize(cv::FileStorage fSettings)
{
    FrameId = 0;

    // Image resolution
    h_width  = 1920;
    h_height = 960;
    std::cout << "[Width,Height]: "<< h_width << "," << h_height << std::endl;

    // Voxel Size
    h_voxelSize = fSettings["VoxelSize"];
    std::cout << "VoxelSize: " << h_voxelSize << std::endl;

    // Truncation
    h_truncation = fSettings["Truncation"];
    std::cout << "Truncation: " << h_truncation << std::endl;

    h_resolution = fSettings["Resolution"];
    std::cout << "Resolution: " << h_resolution << std::endl;

    // Grid Size
    h_gridSize_x = fSettings["GridSizex"];
    h_gridSize_y = fSettings["GridSizey"];
    h_gridSize_z = fSettings["GridSizez"];
    // h_gridSize = 60;
    std::cout << "GridSize: " << h_gridSize_x << " " << h_gridSize_y << " " << h_gridSize_z << std::endl;

    // thresh
    tsdf_thresh = fSettings["tsdf_thresh"];
    weight_thresh = fSettings["weight_thresh"];
    std::cout << "tsdf_thresh: " << tsdf_thresh << std::endl;
    std::cout << "weight_thresh: " << weight_thresh << std::endl;

    std::cout << "Initialize TSDF ..." << std::endl;

    // allocate memory on GPU
    // int3 label
    checkCudaErrors(cudaMalloc(&d_labels, h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(int3_label)));
    // TSDF model
    checkCudaErrors(cudaMalloc(&d_SDFBlocks, h_gridSize_x * h_gridSize_y * h_gridSize_z * 1000 * sizeof(Voxel)));
    // depth data
    checkCudaErrors(cudaMalloc(&d_depth, 512 * 1024 * sizeof(float)));
    // color data
    checkCudaErrors(cudaMalloc(&d_color, h_height * h_width * 3*sizeof(uint8_t)));
    // pose in base coordinates
    checkCudaErrors(cudaMalloc(&d_pose, 4 * 4 * sizeof(double)));
}

// 将数据从CPU复制到GPU上 copy data to gpu
void CUDATSDFIntegrator::uploadData(const float* depth_cpu_data, const uint8_t* color_cpu_data, const double* pose_cpu, unordered_set<int3_label> &vst, unordered_map<int3_label, VoxelBlock*> &vmp)
{
    checkCudaErrors(cudaMemcpy(d_depth, depth_cpu_data, 512 * 1024 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_color, color_cpu_data, h_height * h_width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pose, pose_cpu, 4 * 4 * sizeof(double), cudaMemcpyHostToDevice));

    int num = 0;
    for(unordered_set<int3_label>::iterator it = vst.begin(); it != vst.end(); ++it, num++)
    {
        if(!vmp[*it]) vmp[*it] = new VoxelBlock;
        checkCudaErrors(cudaMemcpy(d_labels+num, &(*it), sizeof(int3_label), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((d_SDFBlocks+1000*num), &(vmp[*it]->voxels[0]), 1000 * sizeof(Voxel), cudaMemcpyHostToDevice));
    }
    FrameNum = num;
}

// Integrate depth and color into TSDF model
void CUDATSDFIntegrator::integrate()
{
    //std::cout << "Fusing color image and depth" << std::endl;

    // Integrate function
    IntegrateDepthMapCUDA(d_labels, d_pose, d_depth, d_color, h_truncation, h_resolution, h_voxelSize, h_height, h_width, h_gridSize_x, h_gridSize_y, h_gridSize_z, d_SDFBlocks, FrameNum);

    FrameId++;
    std::cout << "Frame Index:" << FrameId << std::endl;
}

void CUDATSDFIntegrator::downloadData(unordered_set<int3_label> &vst, unordered_map<int3_label, VoxelBlock*> &vmp)
{
    int num = 0;
    for(unordered_set<int3_label>::iterator it = vst.begin(); it != vst.end(); ++it, num++)
    {
        checkCudaErrors(cudaMemcpy(&(vmp[*it]->voxels[0]), (d_SDFBlocks+1000*num), 1000 * sizeof(Voxel), cudaMemcpyDeviceToHost));
    }
}

// uint8_t getaverage(uint8_t c1, uint8_t c2)
// {
//     if (c1==0) return c2;
//     if (c2==0) return c1;
//     return (uint8_t)floor(c1*0.5 + c2*0.5);
// }

// Compute surface points from TSDF voxel grid and save points to point cloud file
void CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud(unordered_map<int3_label, VoxelBlock*> &vmp)
{
    int delta[8] = {0, 100, 110, 10, 1, 101, 111, 11}; // 八个顶点对应的下标i的增量
    PointCloud::Ptr pointcloud = boost::make_shared< PointCloud >( );
    unordered_map<int3_label, VoxelBlock*>::iterator tempit0;
    unordered_map<int3_label, VoxelBlock*>::iterator tempit1;
    int printcnt = 0;
    for(unordered_map<int3_label, VoxelBlock*>::iterator it = vmp.begin(); it != vmp.end(); ++it)
    {
        printcnt++;
        if(printcnt%10000==0)cout << "save voxel blocks num: " << printcnt << endl;
        for (int i=0;i<1000;i++)
        {
            if (std::abs(it->second->voxels[i].sdf) < tsdf_thresh)
            {
                bool breakflag = false;
                float x = it->first.x * ( h_resolution * h_voxelSize ) + floor(i / 100) * h_voxelSize;
                float y = it->first.y * ( h_resolution * h_voxelSize ) + ((int)floor(i / 10) % 10) * h_voxelSize;
                float z = it->first.z * ( h_resolution * h_voxelSize ) + (i % 10) * h_voxelSize;
                int mcindex = 0;
                int3_label tmplab[8];
                int newindex[8] = {0};
                for (int jj=0;jj<8;jj++)
                {
                    int j = 7-jj;
                    float xx, yy, zz;
                    tmplab[j].x = it->first.x;
                    tmplab[j].y = it->first.y;
                    tmplab[j].z = it->first.z;
                    if (i+delta[j]>=1000)
                    {
                        if (j==1||j==2||j==5||j==6) xx = x + h_voxelSize;
                        if (j==2||j==3||j==6||j==7) yy = y + h_voxelSize;
                        if (j==4||j==5||j==6||j==7) zz = z + h_voxelSize;
                        tmplab[j].x = (int)floor(xx/(h_resolution * h_voxelSize));
                        tmplab[j].y = (int)floor(yy/(h_resolution * h_voxelSize));
                        tmplab[j].z = (int)floor(zz/(h_resolution * h_voxelSize));
                        newindex[j] = ((xx - tmplab[j].x * (h_resolution * h_voxelSize)) / h_voxelSize * 100) + ((yy - tmplab[j].y * (h_resolution * h_voxelSize)) / h_voxelSize * 10) + ((zz - tmplab[j].z * (h_resolution * h_voxelSize)) / h_voxelSize);
                    }
                    else newindex[j] = i+delta[j];
                    tempit0 = vmp.find(tmplab[j]);
                    if (tempit0 == vmp.end())
                    {
                        breakflag = true;
                        break;
                    }
                    if (tempit0->second->voxels[newindex[j]].sdf>=0) mcindex++;
                    if (jj<7)mcindex <<= 1;
                }
                // cout << mcindex << endl;
                if (breakflag) continue;
                int vis[12] = {0};
                for (int k = 0;k<16;k++)
                {
                    int midnum = findtable[mcindex][k];
                    if (midnum==-1) break;
                    if (vis[midnum]==1)continue;
                    vis[midnum] = 1;
                    pcl::PointXYZRGBA point;
                    point.x = x + mid_pos[midnum][0] * h_voxelSize;
                    point.y = z + mid_pos[midnum][2] * h_voxelSize;
                    point.z = -(y + mid_pos[midnum][1] * h_voxelSize);
                    point.r = it->second->voxels[i].r;
                    point.g = it->second->voxels[i].g;
                    point.b = it->second->voxels[i].b;
                    // tempit0 = vmp.find(tmplab[end_point[midnum][0]]);
                    // tempit1 = vmp.find(tmplab[end_point[midnum][1]]);
                    // point.r = getaverage(tempit0->second->voxels[newindex[end_point[midnum][0]]].r, tempit1->second->voxels[newindex[end_point[midnum][1]]].r);
                    // point.g = getaverage(tempit0->second->voxels[newindex[end_point[midnum][0]]].g, tempit1->second->voxels[newindex[end_point[midnum][1]]].g);
                    // point.b = getaverage(tempit0->second->voxels[newindex[end_point[midnum][0]]].b, tempit1->second->voxels[newindex[end_point[midnum][1]]].b);
                    pointcloud->points.push_back(point);
                }
                
                // point.x = x;
                // point.y = z;
                // point.z = -y;
                // point.r = it->second->voxels[i].r;
                // point.g = it->second->voxels[i].g;
                // point.b = it->second->voxels[i].b;
                // pointcloud->points.push_back(point);
            }
        }
        // delete it->second;
    }
    std::cout << pointcloud->points.size() << std::endl;
    pcl::io::savePLYFileBinary("/home/code/web/upload/tsdf_new.ply", *pointcloud);

    for(unordered_map<int3_label, VoxelBlock*>::iterator it = vmp.begin(); it != vmp.end(); ++it)
        delete it->second;
    // FILE *fp = fopen("tsdf_test.ply", "w");
    // fprintf(fp, "ply\n");
    // fprintf(fp, "format binary_little_endian 1.0\n");
    // fprintf(fp, "element vertex %d\n", pointcloud->points.size());
    // fprintf(fp, "property float x\n");
    // fprintf(fp, "property float y\n");
    // fprintf(fp, "property float z\n");
    // fprintf(fp, "property uchar red\n");
    // fprintf(fp, "property uchar green\n");
    // fprintf(fp, "property uchar blue\n");
    // fprintf(fp, "end_header\n");
    // for(int i = 0; i < pointcloud->points.size(); i++)
    // {
    //     float x = pointcloud->points[i].x;
    //     float y = pointcloud->points[i].z;
    //     float z = -pointcloud->points[i].y;
    //     fwrite(&x, sizeof(float), 1, fp);
    //     fwrite(&y, sizeof(float), 1, fp);
    //     fwrite(&z, sizeof(float), 1, fp);
    //     fwrite(&pointcloud->points[i].r, sizeof(uchar), 1, fp);
    //     fwrite(&pointcloud->points[i].g, sizeof(uchar), 1, fp);
    //     fwrite(&pointcloud->points[i].b, sizeof(uchar), 1, fp);
    // }
    // fclose(fp);
}

// Default deconstructor
CUDATSDFIntegrator::~CUDATSDFIntegrator()
{
    checkCudaErrors(cudaFree(d_labels));
    checkCudaErrors(cudaFree(d_SDFBlocks));
    checkCudaErrors(cudaFree(d_depth));
    checkCudaErrors(cudaFree(d_color));
}