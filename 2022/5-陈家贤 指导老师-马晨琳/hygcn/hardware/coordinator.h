//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_COORDINATOR_H
#define DRAMSIM3_COORDINATOR_H
#include <memory>
#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <dram_system.h>
#include "dataflow/event.h"
#include "aggr_buf.h"
#include <hmc.h>

class Coordinator {
public:
    Coordinator(const std::shared_ptr<dramsim3::JedecDRAMSystem>& dram,
                const std::shared_ptr<AggrBuf>& edram):
                dram(dram), edram(edram) {

    }

    void AddEvent(EventPtr event);

    void IssueEvent();

    void ClockTick();


    std::shared_ptr<dramsim3::JedecDRAMSystem> dram;
    std::shared_ptr<AggrBuf> edram;
    std::unordered_multimap<uint64_t, EventPtr> events;
    std::unordered_multimap<uint64_t, EventPtr> edram_events;
    std::deque<EventPtr> edges;
    std::deque<EventPtr> inputs;
    std::deque<EventPtr> weights;
    std::deque<EventPtr> outputs;
    std::deque<EventPtr> tmps;

    uint64_t byte_edge_read = 0;
    uint64_t byte_input_read = 0;
    uint64_t byte_weights_read = 0;
    uint64_t byte_output_write = 0;
    uint64_t byte_aggr_read = 0;
    uint64_t byte_aggr_write = 0;
};

#endif //DRAMSIM3_COORDINATOR_H
