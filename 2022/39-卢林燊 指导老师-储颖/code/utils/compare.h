//
// Created by lj on 2022/11/9.
//

#ifndef FUXIAN_COMPARE_H
#define FUXIAN_COMPARE_H


#include "../entity/view_point.h"
#include "../entity/view_score.h"
#include "../entity/sample_point.h"
class Compare {
public:
    static bool compareViewPoint(const ViewPoint& v1, const ViewPoint& v2);
    static bool compareByViewScoreFromBigToSmall(const ViewScore &viewScore1, const ViewScore &viewScore2);
    static bool spcompareByQuality(const SamplePoint &samplePoint1, const SamplePoint &samplePoint2);
};


#endif //FUXIAN_COMPARE_H
