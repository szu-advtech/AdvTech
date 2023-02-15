//
// Created by lj on 2022/11/7.
//

#ifndef FUXIAN_IO_UTILS_H
#define FUXIAN_IO_UTILS_H

#include "iostream"
#include "vector"
#include "../entity/sample_point.h"
#include "../entity/view_point.h"
#include <fstream>
class IOUtils {
public:
    static void readPlyFile(const std::string& filePath, std::vector<SamplePoint>& points,int type); // type=1 为ply文件包含quality属性
    static void readPlyForAll(const std::string& file, std::vector<SamplePoint> &meshPoint, std::vector<int*> &faces);
    static void saveViewPoint(const std::string &outputDir, const std::string &folderName, const std::string filename,std::vector<ViewPoint> viewpoints );
    static void viewPointsToSmithPath(std::vector<ViewPoint> &viewPoints, std::string &savePath);
};


#endif //FUXIAN_IO_UTILS_H
