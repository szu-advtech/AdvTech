//
// Created by lj on 2022/11/10.
//

#ifndef FUXIAN_MAP_H
#define FUXIAN_MAP_H
#include "iostream"
#include "vector"
#include "sample_point.h"

class Map {
private:
    float xMin;
    float xMax;
    float yMin;
    float yMax;
    float zMin;
    float zMax;
    float xOffset;       // x偏移量
    float yOffset;       // y偏移量
    float xResolution;   // x分辨率
    float yResolution;   // y分辨率
    int mapWidth;
    int mapHeight;
    std::vector<std::vector<float>> heightMap;
    void initBoundary(std::vector<SamplePoint>& points);  // 初始化边界
    void initHeightMap(std::vector<SamplePoint>& points);  // 初始化地图高度
public:
    Map(std::vector<SamplePoint>& points, float xResolution=10.0, float yResolution=10.0);
    float getXOffset();
    float getYOffset();
    int getXResolution();
    int getYResolution();
    int getMapWidth();
    int getMapHeight();
    float getMapData(int row, int col);
    float getMinHeightOfPos(float xPos, float yPos);
};


#endif //FUXIAN_MAP_H
