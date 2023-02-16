//
// Created by lj on 2022/11/10.
//

#include "map.h"

Map::Map(std::vector<SamplePoint> &points, float xResolution, float yResolution) {
    this->xResolution = xResolution;
    this->yResolution = yResolution;
    // 初始化地图边界
    initBoundary(points);
    initHeightMap(points);
}


void Map::initBoundary(std::vector<SamplePoint> &points) {
    this->xMin = FLT_MAX;
    this->xMax = FLT_MIN;
    this->yMin = FLT_MAX;
    this->yMax = FLT_MIN;
    this->zMin = FLT_MAX;
    this->zMax = FLT_MIN;
    for(SamplePoint& sp: points){
        xMin = std::min(sp.getPos()[0], xMin);
        xMax = std::max(sp.getPos()[0], xMax);
        yMin = std::min(sp.getPos()[1], yMin);
        yMax = std::max(sp.getPos()[1], yMax);
        zMin = std::min(sp.getPos()[2], zMin);
        zMax = std::max(sp.getPos()[2], zMax);
    }
    this->xOffset = 0 - this->xMin;
    this->yOffset = 0 - this->yMin;
}

void Map::initHeightMap(std::vector<SamplePoint> &points) {
    float mapWidthRange = this->xMax - this->xMin;
    float mapHeightRange = this->yMax - this->yMin;
    this->mapWidth = ceil(mapWidthRange / this->xResolution)+1;
    this->mapHeight = ceil(mapHeightRange / this->yResolution)+1;
    for(int i=0; i<this->mapWidth; i++){
        std::vector<float> xLine(this->mapHeight, this->zMin);
        this->heightMap.push_back(std::move(xLine));
    }
    // 更新heightMap
    for(SamplePoint& sp: points){
        int row = ceil((sp.getPos()[0] + this->xOffset) / this->xResolution);
        int col = ceil((sp.getPos()[1] + this->yOffset) / this->yResolution);
        this->heightMap[row][col] = std::max(sp.getPos()[2], this->heightMap[row][col]);
    }
}

float Map::getXOffset() {
    return this->xOffset;
}

float Map::getYOffset() {
    return this->yOffset;
}

int Map::getXResolution() {
    return this->xResolution;
}

int Map::getYResolution() {
    return this->yResolution;
}

int Map::getMapWidth() {
    return this->mapWidth;
}

int Map::getMapHeight() {
    return this->mapHeight;
}

float Map::getMapData(int row, int col) {
    return this->heightMap[row][col];
}

float Map::getMinHeightOfPos(float xPos, float yPos) {
    int row = ceil(xPos + xOffset) / xResolution;
    int col = ceil(yPos + yOffset) / yResolution;
    if(row >= 0 && row < mapWidth && col >= 0 && col < mapHeight){
        // 返回高度
        return heightMap[row][col];
    }
    return -1;
}