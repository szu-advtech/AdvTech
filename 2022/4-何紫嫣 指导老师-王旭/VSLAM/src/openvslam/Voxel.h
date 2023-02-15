//
// Created by will on 19-11-15.
//

#pragma once

#ifndef TSDF_VOXEL_H
#define TSDF_VOXEL_H

#include <cuda_runtime.h>
#include <cstring>
#include <functional>

/**
 * @brief Voxel defination
 */


struct int3_label
{
    // 实际体素块左下角坐标*10，需要算坐标时取出后转float除以10即可
    int x,y,z;
    // Default Constructor
    __device__ __host__ int3_label()
    {
        x = 0;
        y = 0;
        z = 0;
    }
    __device__ __host__ void operator=(struct int3_label &v)
    {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
    }
};


namespace std{
    template<>
    struct hash<int3_label>{//哈希的模板定制
    public:
        size_t operator()(const int3_label &p) const 
        {
            return hash<int>()(p.x) ^ hash<int>()(p.y) ^ hash<int>()(p.z);
        }
        
    };
    
    template<>
    struct equal_to<int3_label>{//等比的模板定制
    public:
        bool operator()(const int3_label &p1, const int3_label &p2) const
        {
            return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z;
        }
        
    };
}

struct Voxel
{
    float sdf;
    float weight;
    int num;
    uint8_t b, g, r;
    // uchar4 color;
    //unsigned char color[4];//R,G,B,A

    // Default Constructor
    __device__ __host__ Voxel()
    {
        sdf = 1.0f; // signed distance function
        weight = 0.0f; // accumulated weight
        b = 0;
        g = 0;
        r = 0;
        num = 0;
        // color = make_uchar4(0, 0, 0, 0); //R,G,B
    }

    __device__ __host__ void operator=(struct Voxel &v)
    {
        this->sdf = v.sdf;
        this->weight = v.weight;
        this->num = v.num;
        this->g = v.g;
        this->b = v.b;
        this->r = v.r;
    }
};

struct VoxelBlock
{
    Voxel voxels[1000];
    int lastFrame;
};

#endif //TSDF_VOXEL_H
