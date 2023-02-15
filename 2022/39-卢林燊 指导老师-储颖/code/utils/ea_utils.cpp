//
// Created by lj on 2022/11/9.
//

#include "ea_utils.h"
#include "compare.h"
#include "ply_utils.h"
#include "log.h"


std::default_random_engine EAUtils::e;

void EAUtils::initPopulation(ScoreUtils &scoreUtils,
                             Map &map,int popSize,
                             std::vector<std::vector<ViewPoint>> &population,
                             std::vector<SamplePoint> &points) {

    std::vector<std::vector<ViewPoint>> allViewpoints;
    allViewpoints.resize(points.size(), std::vector<ViewPoint>());
    population.resize(Params::POP_SIZE);

    std::vector<ViewPoint> allviewpoints2,finalviews;
    for(int i=0; i<points.size(); i++){
        Log::info("initialing the view of the " + std::to_string(i+1) + "th samplePoint");
        std::vector<ViewPoint> viewPoints;
        initViews(map,viewPoints, points[i]);
        // 纠正高度
        correctHeight(map,viewPoints, points[i]);  //对每个视角纠正高度
//        std::cout<<std::to_string(i+1) + "th samplePoint's viewpoint nums:  "<<viewPoints.size()<<std::endl;
        // 根据和法线的角度从小到大进行排序
        std::sort(viewPoints.begin(), viewPoints.end(), Compare::compareViewPoint);
        for(int j=0; j<viewPoints.size(); j++){
            allViewpoints[i].push_back(std::move(viewPoints[j]));
            allviewpoints2.push_back(std::move(viewPoints[j]));
        }
    }
    selectViewPointByScore(scoreUtils,map,finalviews,points,allViewpoints,allviewpoints2);

    population[0]=finalviews;
    std::cout<<"population[0] size:"<<population[0].size()<<"***-----"<<std::endl;

}

void EAUtils::normalGrowth(ViewPoint &vp, SamplePoint &samplePoint) {
    float axisSum = abs(vp.getDirection()[0]) + abs(vp.getDirection()[1]) + abs(vp.getDirection()[2]);
    float newPos[3];
    // 计算新的pos[0]
    if(vp.getDirection()[0] == 0){
        newPos[0] = vp.getPos()[0];
    }else if(vp.getDirection()[0] > 0){
        newPos[0] = samplePoint.getPos()[0] - sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[0])/axisSum));
    }else{
        newPos[0] = samplePoint.getPos()[0] + sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[0])/axisSum));
    }
    // 计算新的pos[1]
    if(vp.getDirection()[1] == 0){
        newPos[1] = vp.getPos()[1];
    }else if(vp.getDirection()[1] > 0){
        newPos[1] = samplePoint.getPos()[1] - sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[1])/axisSum));
    }else{
        newPos[1] = samplePoint.getPos()[1] + sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[1])/axisSum));
    }
    // 计算新的pos[2]
    if(vp.getDirection()[2] == 0){
        newPos[2] = vp.getPos()[2];
    }else if(vp.getDirection()[2] > 0){
        newPos[2] = samplePoint.getPos()[2] - sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[2])/axisSum));
    }else{
        newPos[2] = samplePoint.getPos()[2] + sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[2])/axisSum));
    }
    vp.setPos(newPos);
}

void EAUtils::initViews(Map& map,std::vector<ViewPoint> &viewContainer, SamplePoint &sp) {
    // samplepoint  sp
    float *spPos = sp.getPos();
    float *spDirection = sp.getDirection();
    float levelDirection[3] = {spDirection[0], spDirection[1], 0};
    float radian = PlyUtils::calRadianOfPoints(levelDirection, spDirection);

    ViewPoint topView(sp.getPos()[0]+sp.getDirection()[0],
                      sp.getPos()[1]+sp.getDirection()[1],
                      sp.getPos()[2]+sp.getDirection()[2],
                      -sp.getDirection()[0],
                      -sp.getDirection()[1],
                      -sp.getDirection()[2]);
//    topView.angle=-M_PI;
    normalGrowth(topView, sp);
    viewContainer.push_back(topView);

    if(radian >= M_PI/6) {
        // 初始化水平视角
        Log::debug("level init");
        initLevelView(viewContainer, sp);
        // 以高度变化为准
        float totalHeight = abs(topView.getPos()[2] - sp.getPos()[2]);
        if (topView.getPos()[2] >= sp.getPos()[2]) {
            for (float start = topView.getPos()[2] - 0.5; start >= sp.getPos()[2]; start -= 2) {
                float ratio = 1 - (start - sp.getPos()[2]) / totalHeight;
                int initViewNums = ceil(36 * ratio);
                if (initViewNums < 3) {
                    initViewNums = 3;
                }
                int addNum = 36 / initViewNums;
                for (int i = 1; i <= 36; i += addNum) {
                    float pos[3] = {topView.getPos()[0] + ratio * (viewContainer[i].getPos()[0] - topView.getPos()[0]),
                                    topView.getPos()[1] + ratio * (viewContainer[i].getPos()[1] - topView.getPos()[1]),
                                    topView.getPos()[2] + ratio * (viewContainer[i].getPos()[2] - topView.getPos()[2])};
                    float direction[3] = {pos[0] - sp.getDirection()[0],
                                          pos[1] - sp.getDirection()[1],
                                          pos[2] - sp.getDirection()[2]};
                    ViewPoint v(pos, direction);
                    viewContainer.push_back(std::move(v));

                }
            }
        } else {
            for (float start = topView.getPos()[2]; start <= sp.getPos()[2]; start += 2) {
                float ratio = 1 - (sp.getPos()[2] - start) / totalHeight;
                int initViewNums = ceil(36 * ratio);
                if (initViewNums < 3) {
                    initViewNums = 3;
                }
                int addNum = 36 / initViewNums;
                for (int i = 1; i <= 36; i += addNum) {
                    float pos[3] = {topView.getPos()[0] + ratio * (viewContainer[i].getPos()[0] - topView.getPos()[0]),
                                    topView.getPos()[1] + ratio * (viewContainer[i].getPos()[1] - topView.getPos()[1]),
                                    topView.getPos()[2] + ratio * (viewContainer[i].getPos()[2] - topView.getPos()[2])};
                    float direction[3] = {pos[0] - sp.getDirection()[0],
                                          pos[1] - sp.getDirection()[1],
                                          pos[2] - sp.getDirection()[2]};
                    ViewPoint v(pos, direction);
                    viewContainer.push_back(v);
                }
            }
        }
    }else {
        Log::debug("vertical init");
        initVerticalView(viewContainer, sp);
        // 以y变化为准
        float totalHeight = abs(topView.getPos()[1] - sp.getPos()[1]);
        if(topView.getPos()[1] >= sp.getPos()[1]){
            for(float start=topView.getPos()[1]-0.5; start>=sp.getPos()[1]; start-=2){
                float ratio = 1-(start-sp.getPos()[1])/totalHeight; // 1-(视点y方向高度-样本点y方向高度)/y方向最大的高度
                int initViewNums = ceil(36 * ratio);
                if(initViewNums < 3){
                    initViewNums = 3;
                }
                int addNum = 36 / initViewNums;
                for(int i=1; i<=36; i+=addNum){
                    float pos[3] = {topView.getPos()[0]+ratio*(viewContainer[i].getPos()[0]-topView.getPos()[0]),
                                    topView.getPos()[1]+ratio*(viewContainer[i].getPos()[1]-topView.getPos()[1]),
                                    topView.getPos()[2]+ratio*(viewContainer[i].getPos()[2]-topView.getPos()[2])};
                    float direction[3] = {pos[0]-sp.getDirection()[0],
                                          pos[1]-sp.getDirection()[1],
                                          pos[2]-sp.getDirection()[2]};
                    ViewPoint v(pos, direction);
                    viewContainer.push_back(v);
                }
            }
        }else{
            for(float start=topView.getPos()[1]; start<=sp.getPos()[1]; start+=2){
                float ratio = 1-(sp.getPos()[1]-start)/totalHeight;
                int initViewNums = ceil(36 * ratio);
                if(initViewNums < 3){
                    initViewNums = 3;
                }
                int addNum = 36 / initViewNums;
                for(int i=1; i<=36; i+=addNum){
                    float pos[3] = {topView.getPos()[0]+ratio*(viewContainer[i].getPos()[0]-topView.getPos()[0]),
                                    topView.getPos()[1]+ratio*(viewContainer[i].getPos()[1]-topView.getPos()[1]),
                                    topView.getPos()[2]+ratio*(viewContainer[i].getPos()[2]-topView.getPos()[2])};
                    float direction[3] = {pos[0]-sp.getDirection()[0],
                                          pos[1]-sp.getDirection()[1],
                                          pos[2]-sp.getDirection()[2]};
                    ViewPoint v(pos, direction);
                    viewContainer.push_back(v);
                }
            }
        }
    }

}

void EAUtils::initLevelView(std::vector<ViewPoint> &viewContainer, SamplePoint &samplePoint) {
    float *pPos = samplePoint.getPos(); // 视点的世界坐标
    float radian = getRandomFloatNumber(-2 * M_PI, 0);

    for (int i = 0; i < 36; i++) {
        float randD = getRandomFloatNumber(Params::MIN_DISTANCE_BETWEEN_POINT_AND_VIEW,
                                           Params::MAX_DISTANCE_BETWEEN_POINT_AND_VIEW);
        float currentRadian = radian - Params::PER_RADIAN * (float) i;
        //计算视点的当前x坐标 v.x
        float view_x = pPos[0] + randD * cos(currentRadian);
        float view_y = 0;
        if ((currentRadian >= -M_PI && currentRadian <= 0) ||
            (currentRadian >= -3 * M_PI && currentRadian <= -2 * M_PI) ||
            (currentRadian >= -5 * M_PI && currentRadian <= -4 * M_PI)) {
            view_y = pPos[1] - sqrt(pow(randD, 2) - pow(std::min(randD, abs(view_x - pPos[0])), 2));
        } else {
            view_y = pPos[1] + sqrt(pow(randD, 2) - pow(std::min(randD, abs(view_x - pPos[0])), 2));
        }
        //同一高度下
        ViewPoint viewPoint(view_x, view_y, pPos[2], view_x - pPos[0], view_y - pPos[1], 0);
        viewContainer.push_back(viewPoint);
    }

}

void EAUtils::initVerticalView(std::vector<ViewPoint> &viewContainer, SamplePoint &samplePoint) {
    float *pPos = samplePoint.getPos();
    // 获取一个随机弧度值*
    float radian = getRandomFloatNumber(-2*M_PI, 0);

    for(int i=0;i<36;i++){
        float randD = getRandomFloatNumber(Params::MIN_DISTANCE_BETWEEN_POINT_AND_VIEW,
                                           Params::MAX_DISTANCE_BETWEEN_POINT_AND_VIEW);
        float currentRadian = radian - Params::PER_RADIAN * (float)i;
        // 计算视角的x坐标 v.x
        float view_x = pPos[0] + randD * cos(currentRadian);
        float view_z = 0;
        if((currentRadian >= -M_PI && currentRadian <= 0) ||
           (currentRadian >= -3*M_PI && currentRadian <= -2*M_PI) ||
           (currentRadian >= -5*M_PI && currentRadian <= -4*M_PI)){
            view_z = pPos[2] - sqrt(pow(randD, 2) - pow(std::min(randD, abs(view_x - pPos[0])), 2));
        }else{
            view_z = pPos[2] + sqrt(pow(randD, 2) - pow(std::min(randD, abs(view_x - pPos[0])), 2));
        }
        ViewPoint viewPoint(view_x, pPos[1], view_z, view_x-pPos[0], 0, view_z-pPos[2]);
        viewContainer.push_back(viewPoint);

    }
}

//高度修正
void EAUtils::correctHeight(Map &map, std::vector<ViewPoint> &viewpoints, SamplePoint &samplePoint) {
    for(ViewPoint& vp:viewpoints){
        float distance = PlyUtils::calDistanceForPoint(vp, samplePoint);
        if (distance<=25) {
            // 所有分量的和
            float axisSum = abs(vp.getPos()[0]) + abs(vp.getPos()[1]) + abs(vp.getPos()[2]);
            float newPos[3];
            // 计算新的pos[0]
            if (vp.getDirection()[0] == 0) {
                newPos[0] = vp.getPos()[0];
            } else if (vp.getDirection()[0] > 0) {
                newPos[0] = samplePoint.getPos()[0] -
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[0]) / axisSum));
            } else {
                newPos[0] = samplePoint.getPos()[0] +
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[0]) / axisSum));
            }
            // 计算新的pos[1]
            if (vp.getDirection()[1] == 0) {
                newPos[1] = vp.getPos()[1];
            } else if (vp.getDirection()[1] > 0) {
                newPos[1] = samplePoint.getPos()[1] -
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[1]) / axisSum));
            } else {
                newPos[1] = samplePoint.getPos()[1] +
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[1]) / axisSum));
            }
            // 计算新的pos[2]
            if (vp.getDirection()[2] == 0) {
                newPos[2] = vp.getPos()[2];
            } else if (vp.getDirection()[2] > 0) {
                newPos[2] = samplePoint.getPos()[2] -
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[2]) / axisSum));
            } else {
                newPos[2] = samplePoint.getPos()[2] +
                            sqrt(pow(Params::BEST_DISTANCE, 2) * (abs(vp.getDirection()[2]) / axisSum));
            }

            vp.setPos(newPos);
            if (vp.getPos()[2] < samplePoint.getPos()[2]) {
                vp.setDirectionByIndex(2, 0);
            }
        }  // end of dis<25

        correctByHeightMap(map, samplePoint, vp);
        if(vp.getPos()[2] < samplePoint.getPos()[2]){
            vp.setDirectionByIndex(2, 0);
        }

        float direction[3] = {
                    vp.getPos()[0] - samplePoint.getPos()[0],
                    vp.getPos()[1] - samplePoint.getPos()[1],
                    vp.getPos()[2] - samplePoint.getPos()[2]};
        vp.angle = PlyUtils::radian2angle(PlyUtils::calRadianOfPoints(direction, samplePoint.getDirection()));
    }
}

//
void EAUtils::correctByHeightMap(Map &map, SamplePoint &samplePoint, ViewPoint &viewPoint) {
    int row = ceil((viewPoint.getPos()[0] + map.getXOffset()) / map.getXResolution());
    int col = ceil((viewPoint.getPos()[1] + map.getYOffset()) / map.getYResolution());
    if(row >= 0 && row < map.getMapWidth() && col >= 0 && col < map.getMapHeight()){
        // 坐标在范围之内
        if(viewPoint.getPos()[2] < map.getMapData(row, col) + 6){
            viewPoint.setPosByIndex(2, 6);
        }
    }else{
        // 坐标在范围之外，最起码要比最低距离高
        viewPoint.setPosByIndex(2, std::max(12.0f, viewPoint.getPos()[2]));
    }
    viewPoint.setDirection(new float[3]{samplePoint.getPos()[0] - viewPoint.getPos()[0],
                                        samplePoint.getPos()[1] - viewPoint.getPos()[1],
                                        samplePoint.getPos()[2] - viewPoint.getPos()[2]});
}

//随机浮点数
float EAUtils::getRandomFloatNumber(float minValue, float maxValue) {
    // 设置随机数分布
    std::uniform_real_distribution<float> u(minValue, maxValue);
    return u(e);
}

void EAUtils::selectViewPointByScore(ScoreUtils &scoreUtils, Map &map,
                                     std::vector<ViewPoint> &finalViews,
                                     std::vector<SamplePoint> &points,
                                     std::vector<std::vector<ViewPoint>> &allViewpoints,
                                     std::vector<ViewPoint> &viewpoints) {
    // finalviews 最终视点
    // points 采样点
    // allviewpoints 每个采样点视角的集合
    // viewpoints 所有视角的集合
    // 更新可见情况
    scoreUtils.setTrajectory(viewpoints);//设置所有视点
    scoreUtils.updateVisibility();//更新视角可见性集合
    std::cout<<viewpoints.size()<<std::endl;  // 152802 所有生成的初始视角
    int len = points.size();
    float cd,co;
    float dom_2=pow(Params::BEST_DISTANCE - Params::MIN_DISTANCE_BETWEEN_POINT_AND_VIEW,2);


    std::vector<int> finalviewIdx;
    std::vector<ViewScore> finalViewscore;
    int mark[500] = {0};//扫描标记
    for (int i = 0; i < len/2; ++i) {  // len/2代表设的阈值为len/2处的这个采样点的质量分数，小于这个质量分数的点我们记为S_low
//        std::cout<<"sp local view point size  "<<scoreUtils.pointViewVisibilitySet[i].size()<<std::endl;
        std::vector<ViewScore> score1;
        for (int j = 0; j < scoreUtils.pointViewVisibilitySet[i].size(); ++j) {
            int vIdx = scoreUtils.pointViewVisibilitySet[i][j];  //视点的序号
//            std::cout<<vIdx<<" ";
            float dis_ij = PlyUtils::calDistanceForPoint(points[i],viewpoints[vIdx]);
            cd = 1-pow(dis_ij-Params::BEST_DISTANCE,2)/dom_2;
            float ns[3] = {points[i].getDirection()[0],
                           points[i].getDirection()[1],
                           points[i].getDirection()[2]};
            float dij[3] = {viewpoints[vIdx].getPos()[0]-points[i].getPos()[0],
                            viewpoints[vIdx].getPos()[1]-points[i].getPos()[1],
                            viewpoints[vIdx].getPos()[2]-points[i].getPos()[2]};
//            std::cout<<dij[0]<<" "<<dij[1]<<" "<<dij[2]<<" "<<std::endl;
            float dij_len = sqrt(pow(dij[0],2)+pow(dij[1],2)+pow(dij[2],2));
            dij[0] = dij[0]/dij_len;
            dij[1] = dij[1]/dij_len;
            dij[2] = dij[2]/dij_len;
            float nd = ns[0]*dij[0]+ns[1]*dij[1]+ns[2]*dij[2];
            co = exp(-1*pow((1-nd),2));
            ViewScore s(vIdx,i,1*cd*co);
            score1.emplace_back(std::move(s));
        }

//        std::cout<<i<<"th sp  score1 size  "<<score1.size()<<std::endl;
        sort(score1.begin(),score1.end(),Compare::compareByViewScoreFromBigToSmall);  //每个采样点和对应的视点的质量分数大到小排序

        int count1 = 0,count = 0;
        int existViewIdx[2];
        for (int k = 0; k < score1.size(); ++k) {
            if (count==2) break;
//            if (count1==2) break;

            if (std::find(finalviewIdx.begin(),finalviewIdx.end(),score1[k].viewIndex)==finalviewIdx.end()){
                finalviewIdx.push_back(score1[k].viewIndex);  //存放视角的索引
                finalViewscore.push_back(score1[k]);          //存放score，里面包含了spIdx和对应的vpIdx
                count++;
            }  //这一步这里是把分数高的两个点给放入finalviewIdx中，但是这可能会导致视角的冗余，因为同一个视角可能会观察到多个采样点，所以这里面一开始写的不是很合理?

/*            //但是打印count1之后又全是0...
//            if (std::find(finalviewIdx.begin(),finalviewIdx.end(),score1[k].viewIndex)!=finalviewIdx.end()){
//                existViewIdx[count1]=score1[k].viewIndex;
//                count1++;//如果在finalviews里面找到一个就排除加入一个
//            }*/
        }

    }  //end of for local view
    std::cout<<"final local view point size  "<<finalviewIdx.size()<<std::endl;
    std::cout<<" point size  "<<len<<std::endl;
    std::cout<<" ----------------------  "<<std::endl;
    for (int l = 0; l < finalviewIdx.size(); ++l) {
        finalViews.emplace_back(viewpoints[finalviewIdx[l]]);
    }

    //计算localview覆盖范围
    int sssss=0;
    for (int l = 0; l < finalViews.size(); ++l) {
        /*for (int m = 0; m < scoreUtils.viewPointVisibilitySet[l].size(); ++m) {
            std::cout<<"view can see sample points:"<<scoreUtils.viewPointVisibilitySet[l][m]<<"  ";
        }
        sssss++;*/
        std::cout<<"local view can see sample points num:"<<scoreUtils.viewPointVisibilitySet[l].size()<<std::endl;

    }
//    std::cout<<sssss<<" final view size"<<finalViews.size()<<std::endl;

    //有个疑问:样本点的mark标记可否用视角可见性集合大小来代替？前面在进行可见性判断的时候已经对每一个采样点每一个视角进行逐一的可见性判断了
    //这里范围覆盖的计算是判断相邻的样本点是否也在该样本点所对应的视场角观察的范围内，那前面的步骤已经对每一个点每一个视角都进行一一组合计算过一次
    //可见性判断了，应该不用在进行范围覆盖计算了吧？
/**
//     剩下1/5的采样点我们可以先比较视角可见性集合是否大于2，如果大于2在里面取寻找所有视角里面是否包含在finalview里面
//    for (int spIdx = len*4/5; spIdx < len; ++spIdx) {
//       int view_nums = scoreUtils.pointViewVisibilitySet[spIdx].size();
//        if (view_nums>=2){
//            for (int j = 0; j < view_nums; ++j) {
//
//            }
//        }
//    }
*/


    /**
//    std::cout<<"---------------------------------------------------"<<std::endl;
    int i;
    int zero_nums = 0;
    for ( i = 0; i < points.size(); ++i) {
//        std::cout<<scoreUtils.viewPointVisibilitySet[i].size()<<std::endl;
//        std::cout<<scoreUtils.pointViewVisibilitySet[i].size()<<std::endl;
        if (scoreUtils.pointViewVisibilitySet[i].size()==0){
            zero_nums++;
        }
//        std::cout<<allViewpoints[i].size()<<std::endl;
    }
    std::cout<<i<<std::endl;
    std::cout<<"zero_nums:  "<<zero_nums<<std::endl;
    std::cout<<"finsh....";
*/

}