//
// Created by lj on 2022/11/9.
//

#ifndef VIEWCREATOR_VIEW_SCORE_H
#define VIEWCREATOR_VIEW_SCORE_H

class ViewScore{
public:
    int viewIndex = -1;
    int pointIndex = -1;
    float score = 0;
    ViewScore(int viewIndex, float score);
    ViewScore(int viewIndex,int pointIndex, float score);
    // 拷贝构造函数
    ViewScore(const ViewScore &viewScore);
    // 移动拷贝构造函数
    ViewScore(ViewScore &&viewScore);
    ViewScore& operator=(ViewScore &viewScore);
    ViewScore& operator=(ViewScore &&viewScore);
};

#endif //VIEWCREATOR_VIEW_SCORE_H
