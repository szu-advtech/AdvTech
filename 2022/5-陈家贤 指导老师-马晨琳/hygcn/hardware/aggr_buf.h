//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_AGGR_BUF_H
#define DRAMSIM3_AGGR_BUF_H

#include "dram_system.h"
#include <deque>
class AggrBuf {

public:
    AggrBuf(uint64_t cap_edram, std::function<void(uint64_t)> callback): cap_edram(cap_edram), callback(callback) {
        aggr_ptr = ready_ptr = comb_ptr = done_ptr = std::make_pair<int, uint64_t> (0 , 0);
    }
    void Init(int sz_aggr_vector, int sz_block) {
        this->sz_aggr_vector = sz_aggr_vector;
        this->sz_block = sz_block;
    }

    void Sync(int start_vertex) {
        aggr_ptr.first = start_vertex;
        ready_ptr.first = start_vertex;
        done_ptr.first = start_vertex;
        comb_ptr.first = start_vertex;
    }

    bool WillAllocate(int num_vertex) {
        uint64_t space_free = (cap_edram + aggr_ptr.second - done_ptr.second) % (cap_edram + 1);
        if ((num_vertex * sz_aggr_vector) <= space_free) {
            return true;
        }
        return false;
    }

    void StartAggr(int num_vertex) {
        aggr_ptr.second = (cap_edram + aggr_ptr.second + num_vertex * sz_aggr_vector) % (cap_edram);
        aggr_ptr.first += num_vertex;

    }

    void ConfirmAggr(int num_vertex) {
        ready_ptr.second = (cap_edram + ready_ptr.second + num_vertex * sz_aggr_vector) % (cap_edram);
        ready_ptr.first += num_vertex;
//        StartComb(num_vertex);
//        ConfirmComb(num_vertex);

    }

    int GetReadyVertexNum() {
        return ready_ptr.first - comb_ptr.first;
    }

    void StartComb(int num_vertex) {
        comb_ptr.second = (cap_edram + comb_ptr.second + num_vertex * sz_aggr_vector) % (cap_edram);
        comb_ptr.first += num_vertex;
    }

    void ConfirmComb(int num_vertex) {
        done_ptr.second = (cap_edram + done_ptr.second + num_vertex * sz_aggr_vector) % (cap_edram);
        done_ptr.first += num_vertex;
    }

    uint64_t GetAggrWriteAddr(int vertex_id, int part_id) {
        assert(vertex_id >= ready_ptr.first);
        uint64_t res = (ready_ptr.second + (vertex_id - ready_ptr.first) * sz_aggr_vector + part_id * sz_block) % (cap_edram);
        assert(res < cap_edram);
        return res;
    }

    void ClockTick() {
        while(!trans.empty()) {
            uint64_t addr = trans.front();
            trans.pop_front();
            callback(addr);
        }
    }

    bool WillAcceptTransaction(uint64_t addr, bool is_write) {
        return trans.size() < trans_per_clock;
    }

    bool AddTransaction(uint64_t addr, bool is_write) {
        trans.emplace_back(addr);
        return true;
    }

    const int trans_per_clock = 2;
    std::deque<uint64_t> trans;

    std::pair<int, uint64_t> aggr_ptr;
    std::pair<int, uint64_t> ready_ptr;
    std::pair<int, uint64_t> comb_ptr;
    std::pair<int, uint64_t> done_ptr;

    int sz_aggr_vector = -1;
    int sz_block = -1;
    const uint64_t cap_edram;
    std::function<void(uint64_t)> callback;

};
#endif //DRAMSIM3_AGGR_BUF_H
