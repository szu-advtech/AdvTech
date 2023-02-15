//
// Created by lj on 2022/11/9.
//

#include "compare.h"

bool Compare::compareViewPoint(const ViewPoint &v1, const ViewPoint &v2) {
    if(v1.angle < v2.angle){
        return true;
    }
    return false;
}

bool Compare::compareByViewScoreFromBigToSmall(const ViewScore &viewScore1, const ViewScore &viewScore2) {
    if(viewScore1.score > viewScore2.score){
        return true;
    }
    return false;
}

bool Compare::spcompareByQuality(const SamplePoint &samplePoint1, const SamplePoint &samplePoint2) {
    if (samplePoint1.quality>samplePoint2.quality){
        return true;
    }
    return false;
}