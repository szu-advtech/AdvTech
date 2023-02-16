//
// Created by lj on 2022/11/7.
//

#include "io_utils.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <io.h>
#include <direct.h>
#include <cmath>
#include "tinyply.h"
#include "tools.h"
using namespace std;
#define M_PI 3.1415926
// 比较算子
bool cmp(const SamplePoint &point1,const SamplePoint &point2){
    return point1.quality<point2.quality;
}

void IOUtils::readPlyFile(const std::string &filePath, std::vector<SamplePoint> &points, int type) {
    std::ifstream file(filePath, std::ios::binary);
    if(file.fail()){
        throw std::runtime_error("failed to open" + filePath);
    }
    tinyply::PlyFile plyFile;
    plyFile.parse_header(file);
    std::shared_ptr<tinyply::PlyData> info;
    try{
        if (type == 1)
            info = plyFile.request_properties_from_element("vertex", {"x", "y", "z", "nx", "ny", "nz","quality"});
        else if(type == 2)
            info = plyFile.request_properties_from_element("vertex", {"x", "y", "z", "nx", "ny", "nz"});
    }catch (const std::exception& e){
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
    plyFile.read(file);
    if(info){
        if (type==1){
            float *data = (float*)info->buffer.get();
            int maxSize = info->count * 7;
            for(int i=0; i<maxSize; i+=7){
                SamplePoint samplePoint(data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6]);
                // 调用拷贝构造函数
                points.push_back(std::move(samplePoint));
            }
            //质量排序,根据采样点的质量进行排序
            sort(points.begin(),points.end(),cmp);
        } else if (type==2){
            float *data = (float*)info->buffer.get();
            int maxSize = info->count * 6;
            for(int i=0; i<maxSize; i+=6){
                SamplePoint samplePoint(data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5]);
                // 调用拷贝构造函数
                points.push_back(std::move(samplePoint));
            }
        }
    }else{
        throw std::runtime_error("IOUtils: empty file");
    }
    file.close();
}
void IOUtils::readPlyForAll(const std::string& filePath, std::vector<SamplePoint> &meshPoint, std::vector<int*> &faceIndexes) {
    // 先读取vertex
    readPlyFile(filePath, meshPoint,2);
    // 再读取face
    std::ifstream file(filePath, std::ios::binary);
    if(file.fail()){
        throw std::runtime_error("failed to open" + filePath);
    }
    tinyply::PlyFile plyFile;
    plyFile.parse_header(file);

    std::shared_ptr<tinyply::PlyData> info;
    try{
        info = plyFile.request_properties_from_element("face", {"vertex_indices"}, 3);
    }catch (const std::exception& e){
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
    plyFile.read(file);
    if(info){
        int *data = (int*)info->buffer.get();
        int maxSize = info->count * 3;
        for(int i=0; i<maxSize; i+=3){
            int *face = new int[3]{data[i], data[i+1], data[i+2]};
            faceIndexes.push_back(face);
        }
    }else{
        throw std::runtime_error("IOUtils: empty file");
    }
    file.close();
}

void IOUtils::saveViewPoint(const std::string &outputDir, const std::string &folderName,const std::string filename,
                            std::vector<ViewPoint> viewpoints) {
    // 构造保存路径
//    std::string savePath = outputDir + "\\" + folderName +  "\\" + std::to_string(i+1) + ".txt";
    std::string savePath = outputDir + "\\" + folderName +  "\\" + filename + ".txt";
    if(_access((outputDir + "\\" + folderName).c_str(), 0) == -1){
        _mkdir((outputDir + "\\" + folderName).c_str());
    }
    std::ofstream outputFile(savePath);
    if(outputFile.fail()){
        std::cerr << "[IOUtils::savePopulation] open file fail!" << std::endl;
        return;
    }
    for(int j=0; j<viewpoints.size(); j++){
//        std::cout<<j<<std::endl;
        // 存储每一个视角信息
        outputFile << viewpoints[j].getPos()[0] << ","
                   << viewpoints[j].getPos()[1] << ","
                   << viewpoints[j].getPos()[2] << ","
                   << viewpoints[j].getDirection()[0] << ","
                   << viewpoints[j].getDirection()[1] << ","
                   << viewpoints[j].getDirection()[2] << "\n";
    }
    outputFile.close();

}

void IOUtils::viewPointsToSmithPath(std::vector<ViewPoint> &viewPoints, std::string &savePath) {
    std::string path = Tools::makePath(savePath, "trajectory", "smithFormatPathXueXiao-2.txt");
    std::ofstream file(path);
    if(file.fail()){
        std::cerr << "can't open file" << std::endl;
        exit(-1);
    }
    for(int i=0; i<viewPoints.size(); i++){
        file << setw(4) << setfill('0') << i << ",";
        file << setprecision(8);
        file << -viewPoints[i].getPos()[0] * 100 << ",";  // 将米转换成厘米单位
        file << viewPoints[i].getPos()[1] * 100 << ",";
        file << viewPoints[i].getPos()[2] * 100 << ",";
        float pitch = (atan2(viewPoints[i].getDirection()[2], sqrt(viewPoints[i].getDirection()[0] * viewPoints[i].getDirection()[0] +viewPoints[i].getDirection()[1] * viewPoints[i].getDirection()[1])) * 180.0) / M_PI;
        float yaw = (atan2(viewPoints[i].getDirection()[1], viewPoints[i].getDirection()[0]) * 180.0) / M_PI;
        yaw = 90 - yaw;
        file << -pitch << ",";
        file << 0 << ",";
        file << yaw << "\n";
    }
    file.close();
}

