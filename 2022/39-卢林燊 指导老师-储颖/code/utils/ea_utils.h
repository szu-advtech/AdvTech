//
// Created by lj on 2022/11/9.
//

#ifndef FUXIAN_EA_UTILS_H
#define FUXIAN_EA_UTILS_H

#include <random>
#include "score.h"
#include "Params.h"
#include "../entity/view_point.h"
#include "../entity/sample_point.h"
#include "../entity/map.h"

class EAUtils {
public:
    static std::default_random_engine e;
    static void initPopulation(ScoreUtils &scoreUtils,
                               Map &map,
                               int popSize,
                               std::vector<std::vector<ViewPoint>> &population,
                               std::vector<SamplePoint>& points);
    static void initViews(Map & map, std::vector<ViewPoint>& viewContainer, SamplePoint& samplePoint);

    static void initLevelView(std::vector<ViewPoint> &viewContainer, SamplePoint& samplePoint);
    static void initVerticalView(std::vector<ViewPoint> &viewContainer, SamplePoint& samplePoint);

    static void normalGrowth(ViewPoint &viewPoint, SamplePoint &samplePoint);

    static float getRandomFloatNumber(float minValue, float maxValue);

    static void correctHeight(Map &map, std::vector<ViewPoint> &viewpoints, SamplePoint& samplePoint);
    static void correctByHeightMap(Map &map, SamplePoint& samplePoint, ViewPoint& viewPoint);

    static void selectViewPointByScore(ScoreUtils &scoreUtils,Map &map,
                                       std::vector<ViewPoint> &finalViews,
                                       std::vector<SamplePoint>&points,
                                       std::vector<std::vector<ViewPoint>> &allViewpoints,
                                       std::vector<ViewPoint> &viewpoints);

};


#endif //FUXIAN_EA_UTILS_H
