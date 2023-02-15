//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_SPM_H
#define DRAMSIM3_SPM_H

#include <vector>
#include <iostream>
#include <set>
#include <memory>

class SPMA {
public:
    enum class State{
        FILLING,
        READY,
        DRAINING,
        WRITING,
        IDLE
    };

    SPMA(int cap_buf): cap_buf_(cap_buf) {
        state = State::IDLE;
        event_cnt_ = 0;
    }

    void SwitchState(State s) {
        state = s;
        if(state == State::IDLE) {
            start_vertex = -1;
            end_vertex = -1;

            // edge buf
            edge_chunk_id = 1;
            edge_part_cnt = 0;
            num_edge_part = -1;
            edge_parts.clear();

            // input buf
            neighbors.clear();
            ebuf_ptr = -1;
            input_edge.clear();
            vertex_ptr = -1;
            feature_chunk_cnt = 0;
            num_feature_chunk = 0;
            non_empty_vertex = 0;
        } else if(state == State::READY) {
            vertex_ptr = start_vertex;
        }
    }

    State GetState() const {
        return state;
    }

    bool IsReady() {
        return state == State::READY;
    }

    void AddEvents(int num) {
        event_cnt_ = num;
    }

    void ConfirmEvent() {
        event_cnt_--;
    }

    bool EventDone() {
        return event_cnt_ == 0;
    }

    void SetMatrixShape(int h, int w) {
        matrix_h = h;
        matrix_w = w;
    }

    int matrix_h = 0;
    int matrix_w = 0;


    int start_vertex = -1;
    int end_vertex = -1;

    // some special info, which is implemented ugly;
    // edge buf
    int edge_chunk_id = 1;
    int edge_part_cnt = 0;
    int num_edge_part = -1;

    std::vector<std::vector<std::vector<int>>> edge_parts;

    // input buf
    std::vector<std::vector<int>> input_edge;
    std::set<int> neighbors;
    int ebuf_ptr = -1;
    int vertex_ptr = -1;
    int feature_chunk_cnt = 0;
    int num_feature_chunk = 0;
    int non_empty_vertex = 0;


private:
    State state;
    int cap_buf_;
    int event_cnt_;


};


class SPM {
public:
    enum class State{
        FILLING,
        READY,
        DRAINING,
        WRITING,
        IDLE
    };

    SPM(int cap_buf): cap_buf_(cap_buf) {
        state = State::IDLE;
        event_cnt_ = 0;
    }

    void SwitchState(State s) {
        state = s;
    }

    State GetState() const {
        return state;
    }

    bool IsReady() {
        return state == State::READY;
    }

    void AddEvents(int num) {
        event_cnt_ = num;
    }

    void ConfirmEvent() {
        event_cnt_--;
    }

    bool EventDone() {
        return event_cnt_ == 0;
    }

    void SetMatrixShape(int h, int w) {
        matrix_h = h;
        matrix_w = w;
    }

    int matrix_h;
    int matrix_w;
private:
    State state;
    int cap_buf_;
    int event_cnt_;


};


int GetTargetBuf(const std::vector<std::shared_ptr<SPMA>>& bufs, SPMA::State state);
bool AllBufIdle(const std::vector<std::shared_ptr<SPMA>>& bufs);
#endif //DRAMSIM3_SPM_H
