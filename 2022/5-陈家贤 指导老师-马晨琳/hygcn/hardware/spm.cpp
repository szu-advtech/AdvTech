//
// Created by szu on 2021/11/17.
//
#include "hardware/spm.h"



int GetTargetBuf(const std::vector<std::shared_ptr<SPMA>>& bufs, SPMA::State state) {
    for(int i = 0; i < bufs.size(); i++) {
        if(bufs[i]->GetState() == state) {
            return i;
        }
    }
    return -1;
}

bool AllBufIdle(const std::vector<std::shared_ptr<SPMA>>& bufs) {
    bool idle = true;
    for(int i = 0; i < bufs.size(); i++) {
        if(bufs[i]->GetState() != SPMA::State::IDLE) {
            idle = false;
            break;
        }
    }
    return idle;
}