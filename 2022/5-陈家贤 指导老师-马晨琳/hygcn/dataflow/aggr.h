//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_AGGR_H
#define DRAMSIM3_AGGR_H

#include "config.h"
#include <memory>
#include "hardware/simd.h"
#include "hardware/coordinator.h"
#include "hardware/aggr_buf.h"
#include "hardware/spm.h"
#include "event.h"

class Aggr{
public:
    Aggr (const std::shared_ptr<HyConfig>& hy_config,
          const std::shared_ptr<AggrBuf>& aggr_buf,
          const std::shared_ptr<Coordinator>& coordinator);
    void ClockTick();
    void GenDataFlow();
    void HandleReturnEvent(EventPtr event);
    int GetIdleSIMD();
    std::vector<std::vector<std::vector<int>>> Partition(const std::vector<std::vector<int>>& edges);
    std::set<int> SparsityElimination(std::vector<std::vector<int>> edges);

    int edge_chunk_cnt;
    int num_edge_chunk;

    bool is_done;
    std::vector<std::shared_ptr<SPMA>> ebuf;
    std::vector<std::shared_ptr<SPMA>> inbuf;
    std::shared_ptr<AggrBuf> aggr_buf;
    std::shared_ptr<HyConfig> hy_config;
    std::shared_ptr<Coordinator> coordinator;
    std::vector<std::shared_ptr<SIMD>> simds;
    uint64_t event_out = 0;
    uint64_t event_in = 0;
    uint64_t clk = 0;
    uint64_t end_clk = 0;

};

#endif //DRAMSIM3_AGGR_H
