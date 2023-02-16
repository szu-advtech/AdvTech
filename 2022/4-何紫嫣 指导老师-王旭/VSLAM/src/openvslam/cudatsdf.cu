#include "openvslam/Voxel.h"
#include "openvslam/CUDATSDFIntegrator.h"
#include <cmath>

// CUDA kernel function to integrate a TSDF voxel volume given depth images and color images
__global__ void IntegrateDepthMapKernel(int3_label* d_labels, double* d_pose, float* d_depth, uint8_t* d_color, 
                                        float truncation, float resolution, float voxelSize, int height, int width, Voxel* d_SDFBlocks, int num)
{
    float PI = 3.14159265358979323846f;
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId>=num) return;
    for (int i=0;i<10;i++)
    {
        float xx = d_labels[threadId].x * (resolution*voxelSize) + i * voxelSize;
        for (int j=0;j<10;j++)
        {
            float yy = d_labels[threadId].y * (resolution*voxelSize) + j * voxelSize;
            for (int k=0;k<10;k++)
            {
                float zz = d_labels[threadId].z * (resolution*voxelSize) + k * voxelSize;
                float cam_pt_x = d_pose[0 * 4 + 0] * xx + d_pose[1 * 4 + 0] * yy + d_pose[2 * 4 + 0] * zz + d_pose[3 * 4 + 0];
                float cam_pt_y = d_pose[0 * 4 + 1] * xx + d_pose[1 * 4 + 1] * yy + d_pose[2 * 4 + 1] * zz + d_pose[3 * 4 + 1];
                float cam_pt_z = d_pose[0 * 4 + 2] * xx + d_pose[1 * 4 + 2] * yy + d_pose[2 * 4 + 2] * zz + d_pose[3 * 4 + 2];
                float r = sqrt(cam_pt_x*cam_pt_x+cam_pt_y*cam_pt_y+cam_pt_z*cam_pt_z);
                float n = atan(cam_pt_y/sqrt(cam_pt_x*cam_pt_x+cam_pt_z*cam_pt_z));
                float m = atan(cam_pt_x/cam_pt_z);
                if (cam_pt_z>0) m += PI;
                else if (m<0) m += (2*PI); 
                else m = m;

                // 根据体素坐标，找到图像上对应的像素点pt_pix xy
                int pt_pix_x = m / (2* PI) * width - 0.5;
                int pt_pix_y = (n / PI +0.5) * height - 0.5;
                if(pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                    continue;

                float dist = 1.0f;
                float weight_delta = 1.0f;
                if (r > 0.3)
                {
                    // float depth_val = d_depth[pt_pix_y * width + pt_pix_x]; // 取点对应的深度值
                    float x = pt_pix_x *1.0 / 1.875;
                    float y = pt_pix_y * 1.0 / 1.875;
                    //四个临近点的坐标 (x1,y1)、(x1,y2),(x2,y1)，(x2,y2)
                    int x1,x2;
                    int y1,y2;

                    //两个差值的中值
                    float f12,f34;
                    float	epsilon = 0.0001;

                    //四个临近像素坐标x像素值
                    float f1,f2,f3,f4, ansf;

                    //计算四个临近坐标
                    x1 = (int)x;
                    x2 = x1 + 1;
                    y1 = (int)y;
                    y2 = y1+1;

                    if((x < 0) || (x > 1024 - 1) || (y < 0) || (y > 512 - 1))
                    {
                        ansf = 0.0f;
                    }else{
                        if(fabs(x - 1024+1)<=epsilon) //如果计算点在右测边缘
                        {
                            //如果差值点在图像的最右下角
                            if(fabs(y - 512+1)<=epsilon)
                            {
                                f1 = d_depth[y1 * 1024 + x1];
                                ansf = f1;
                            }else {
                                f1 = d_depth[y1 * 1024 + x1];
                                f3 = d_depth[y2 * 1024 + x1];

                                //图像右方的插值
                                ansf = (f1 + (y-y1)*(f3-f1));
                            }
                        }
                        //如果插入点在图像的下方
                        else if(fabs(y - 512+1)<=epsilon){
                        f1 = d_depth[y1 * 1024 + x1];
                        f2 = d_depth[y1 * 1024 + x2];

                        //图像下方的插值
                        ansf = (f1 + (x-x1)*(f2-f1));
                        }
                        else {
                            //得计算四个临近点像素值
                            f1 = d_depth[y1 * 1024 + x1];
                            f2 = d_depth[y1 * 1024 + x2];
                            f3 = d_depth[y2 * 1024 + x1];
                            f4 = d_depth[y2 * 1024 + x2];

                            //第一次插值
                            f12 = f1 + (x-x1)*(f2-f1); //f(x,0)

                            //第二次插值
                            f34 = f3 + (x-x1)*(f4-f3); //f(x,1)

                            //最终插值
                            ansf = (f12 + (y-y1)*(f34-f12));
                        }
                    }

                    float depth_val = ansf;
                    if(depth_val <= 0 || depth_val > 5) continue;
                    if(sqrt(cam_pt_x*cam_pt_x+cam_pt_z*cam_pt_z)<=0.3) continue;
                    depth_val = depth_val * 0.5;
                    float diff = depth_val - r;
                    if(diff <= -truncation)
                        continue;
                    dist = fmin(1.0f, diff / truncation);
                    // if (diff > 0)
                    // {
                    //     if (r<0.8) weight_delta = 3.0f;
                    //     else if (r < 1.6) weight_delta = 2.0f;
                    //     else weight_delta = 1.0f;
                    // }
                    // else weight_delta = 2.0f;
                    if (r<0.8) weight_delta = 3.0f;
                    else if (r < 1.6) weight_delta = 2.0f;
                    else weight_delta = 1.0f;
                }
                else weight_delta = 50.0f;

                // Integrate TSDF
                int volume_idx = threadId * 1000 + i * 100 + j * 10 + k;
                float weight_old = d_SDFBlocks[volume_idx].weight;
                float num_old = d_SDFBlocks[volume_idx].num;
                // float weight_delta = 0.3 / depth_val;
                float weight_new = weight_old + weight_delta;
                // float weight_new = weight_old + 1.0f;
                d_SDFBlocks[volume_idx].weight = weight_new;
                d_SDFBlocks[volume_idx].num = num_old + 1;
                d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + weight_delta * dist) / weight_new;
                // d_SDFBlocks[volume_idx].sdf = (d_SDFBlocks[volume_idx].sdf * weight_old + dist) / weight_new;

                // Integrate Color
                d_SDFBlocks[volume_idx].b = (d_SDFBlocks[volume_idx].b * num_old + d_color[(pt_pix_y * width + pt_pix_x)*3])/(num_old + 1);
                d_SDFBlocks[volume_idx].g = (d_SDFBlocks[volume_idx].g * num_old + d_color[(pt_pix_y * width + pt_pix_x)*3+1])/(num_old + 1);
                d_SDFBlocks[volume_idx].r = (d_SDFBlocks[volume_idx].r * num_old + d_color[(pt_pix_y * width + pt_pix_x)*3+2])/(num_old + 1);
                // d_SDFBlocks[volume_idx].b = (d_SDFBlocks[volume_idx].b * weight_old + weight_delta * d_color[(pt_pix_y * width + pt_pix_x)*3])/weight_new;
                // d_SDFBlocks[volume_idx].g = (d_SDFBlocks[volume_idx].g * weight_old + weight_delta * d_color[(pt_pix_y * width + pt_pix_x)*3+1])/weight_new;
                // d_SDFBlocks[volume_idx].r = (d_SDFBlocks[volume_idx].r * weight_old + weight_delta * d_color[(pt_pix_y * width + pt_pix_x)*3+2])/weight_new;
            }
        }
    }
}

extern "C" void IntegrateDepthMapCUDA(int3_label* d_labels, double* d_pose, float* d_depth, uint8_t* d_color, 
                                      float truncation, float resolution, float voxelSize, int height, int width, int grid_dim_x, int grid_dim_y, int grid_dim_z, Voxel* d_SDFBlocks, int num)
{
   
    // const dim3 gridSize(grid_dim);
    // const dim3 blockSize(grid_dim, grid_dim);
    const dim3 gridSize(grid_dim_x);
    const dim3 blockSize(grid_dim_y, grid_dim_z);

    std::cout << "Launch Kernel..." << std::endl;
    IntegrateDepthMapKernel <<< gridSize, blockSize >>> (d_labels, d_pose, d_depth, d_color, 
                                                         truncation, resolution, voxelSize, height, width, d_SDFBlocks, num); 

    //cudaError_t status = cudaGetLastError();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}