#include <iostream>
#include <Eigen/Geometry>
#include <vector>
#include "entity/sample_point.h"
#include "utils/io_utils.h"
#include "utils/Params.h"
#include "math.h"
#include "utils/tools.h"
#include "utils/score.h"
#include "utils/log.h"
#include "entity/map.h"
#include "utils/ea_utils.h"
#include <CGAL/Point_set_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <algorithm>
#include "utils/compare.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> SurfaceMesh;
typedef CGAL::Point_set_3<Point_3> PointSet;
typedef SurfaceMesh::Vertex_range VertexRange;
typedef SurfaceMesh::Vertex_index VertexIndex;
typedef SurfaceMesh::Face_index FaceIndex;


int main() {
    std::cout << "Hello, World!" << std::endl;
    //读取点云数据
    std::vector<SamplePoint> samplepoints;
    IOUtils::readPlyFile(Params::SAMPLE_FILE_PATH,samplepoints,2);
//    IOUtils::readPlyFile(Params::SAMPLE_FILE_PATH,samplepoints,1);
//    std::sort(samplepoints.begin(),samplepoints.end(),Compare::spcompareByQuality);
    int len = samplepoints.size();
    float threshold = samplepoints[len*4/5].quality;
//
    std::vector<SamplePoint> meshPoints;  // 存储mesh模型中的所有vertex
    std::vector<int*> faceIndexes;    // 存储每个face的三个顶点序号
    IOUtils::readPlyForAll(Params::MESH_FILE_PATH, meshPoints, faceIndexes);

    ScoreUtils scoreUtils(faceIndexes, meshPoints, samplepoints);
    Map map(samplepoints);
    Log::setLog(true, true, false);
    std::vector<std::vector<ViewPoint>> population;
    std::vector<std::vector<float>> pointScoreInfo;  // 每个个体中每个视角的分数信息
    pointScoreInfo.resize(Params::POP_SIZE, std::vector<float>(samplepoints.size(), 0));

    std::vector<std::vector<std::vector<int>>> pointViewPopInfo; // 存储每个viewpoint可以看到的point序号 => [popIndex][viewIndex][pointIndex];
    pointViewPopInfo.resize(Params::POP_SIZE, std::vector<std::vector<int>>(samplepoints.size(), std::vector<int>()));

    std::vector<std::vector<std::vector<int>>> viewPointPopInfo;        // 存储每个point可以被viewpoint看到的序号 => [popIndex][pointIndex][viewIndex];
    viewPointPopInfo.resize(Params::POP_SIZE, std::vector<std::vector<int>>(Params::TOTAL_VIEW_NUMS, std::vector<int>()));

    std::cout<<scoreUtils.points.size();
    //front bingo
    EAUtils::initPopulation(scoreUtils, map, Params::POP_SIZE, population, samplepoints);
    std::cout<<population[0].size()<<"******"<<std::endl;
    IOUtils::saveViewPoint("D:\\fuxianresult","20230130","xuexiao2",population[0]);
    std::string savepath = "D:\\fuxianresult";
    IOUtils::viewPointsToSmithPath(population[0],savepath);
    std::cout<<" ";
    return 0;
}
