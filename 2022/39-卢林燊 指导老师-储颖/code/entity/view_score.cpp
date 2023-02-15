//
// Created by lj on 2022/11/9.
//
#include "view_score.h"

ViewScore::ViewScore(int viewIndex, float score) {
    this->viewIndex = viewIndex;
    this->score = score;
}

ViewScore::ViewScore(int viewIndex,int pointIndex, float score) {
    this->viewIndex = viewIndex;
    this->score = score;
    this->pointIndex=pointIndex;
}


ViewScore::ViewScore(const ViewScore& viewScore) {
    this->score = viewScore.score;
    this->viewIndex = viewScore.viewIndex;
    this->pointIndex=viewScore.pointIndex;
}

ViewScore::ViewScore(ViewScore &&viewScore) {
    this->score = viewScore.score;
    this->viewIndex = viewScore.viewIndex;
    this->pointIndex=viewScore.pointIndex;
}

ViewScore &ViewScore::operator=(ViewScore &viewScore) {
    this->score = viewScore.score;
    this->viewIndex = viewScore.viewIndex;
    this->pointIndex=viewScore.pointIndex;
    return *this;
}

ViewScore &ViewScore::operator=(ViewScore &&viewScore) {
    this->score = viewScore.score;
    this->viewIndex = viewScore.viewIndex;
    this->pointIndex=viewScore.pointIndex;
    return *this;
}