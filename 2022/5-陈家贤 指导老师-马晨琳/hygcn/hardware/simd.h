//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_SIMD_H
#define DRAMSIM3_SIMD_H

#include <cassert>
#include <iostream>

class SIMD {
public:
    enum class State{
        COMPUTING,
        READY,
        SYNC,
        IDLE
    };

    SIMD(int num_mac, int bit_width_=32):
            num_mac_(num_mac),
            bit_width_(bit_width_),
            writeback_byte(num_mac * bit_width_ / 8){
        assert(bit_width_ % 8 == 0);
        depth = 0;
        width = 0;
        state = State::IDLE;

    }

    void ClockTick() {
        if (state == State::COMPUTING) {
            depth--;
            num_mac_op += width;
            byte_inbuf_read += width * bit_width_ / 8;
            if (depth <= 0) {
                depth = 0;
                width = 0;
                state = State::READY;
            }
        }
    }

    bool AssignTask(int len_feature, int edges) {
       if(state != State::IDLE) {
           return false;
       }
       assert(len_feature <= num_mac_);
       depth = edges;
       width = len_feature;
       state = State::COMPUTING;
       byte_aggrbuf_read += width * bit_width_ / 8;
       byte_aggrbuf_write += width * bit_width_ / 8;
       byte_ebuf_read += depth * bit_width_ / 8;
       return true;
    }

    void SwitchState(State new_state) {
        state = new_state;
        if(state == State::IDLE) {
            vertex_id = -1;
            feature_chunk_id = -1;
            inbuf_ptr = -1;
        }
    }

    bool IsReady() {
        return state == State::READY;
    }

    bool IsSync() {
        return state == State::SYNC;
    }

    bool IsIdle() {
        return state == State::IDLE;
    }


    uint64_t num_mac_op = 0;
    uint64_t byte_inbuf_read = 0;
    uint64_t byte_aggrbuf_read = 0;
    uint64_t byte_aggrbuf_write = 0;
    uint64_t byte_ebuf_read = 0;

    int vertex_id = -1;
    int feature_chunk_id = -1;
    int inbuf_ptr = -1;

    const int writeback_byte;

private:

    const int num_mac_;
    const int bit_width_;
    int depth;
    int width;
    State state;

};
#endif //DRAMSIM3_SIMD_H
