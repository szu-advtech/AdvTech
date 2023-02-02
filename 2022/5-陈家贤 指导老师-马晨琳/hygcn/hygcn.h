//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_HYGCN_H
#define DRAMSIM3_HYGCN_H
#include <iostream>
#include <memory>
#include <vector>
#include <deque>
#include <hmc.h>
#include "dataflow/hy_rec.h"

#include "dataflow/event.h"
#include "hardware/coordinator.h"
#include "dataflow/comb.h"
#include "hardware/aggr_buf.h"
#include "configuration.h"
#include "dram_system.h"
#include "dataflow/aggr.h"
#include "config.h"

class HyGCN {
public:
    HyGCN(const std::shared_ptr<HyConfig>& hy_config);

    void ClockTick();

    void HandleReturnEvent();

    void MemCallback(uint64_t addr) {
        auto iter = coordinator->events.find(addr);
        assert(iter != coordinator->events.end());
        auto event = iter->second;
        event->ConfirmTrans();
        if(event->AllTransConfirmed()) {
            if(event->dir_type == EventDirType::DRAM) {
                dram_return_queue.emplace_back(event);
            } else {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        }
        coordinator->events.erase(iter);
        return;
    }

    void BufCallback(uint64_t addr) {
        auto iter = coordinator->edram_events.find(addr);
        assert(iter != coordinator->edram_events.end());
        auto event = iter->second;
        event->ConfirmTrans();
        if(event->AllTransConfirmed()) {
            if(event->dir_type == EventDirType::EDRAM) {
                edram_return_queue.emplace_back(event);
            } else {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        }
        coordinator->edram_events.erase(iter);
        return;
    }

    bool IsDone() {
        return aggr->is_done && comb->is_done;
        // return comb->IsDone();
    }

    void Record() {
        hyrec.finish_time = clk;
        hyrec.aggr_end_time = aggr->end_clk;
        hyrec.comb_end_time = comb->end_clk;

        for(auto &simd : aggr->simds) {
            hyrec.simd_mac_op += simd->num_mac_op;
            hyrec.sram_ebuf_read += simd->byte_ebuf_read;
            hyrec.sram_inbuf_read += simd->byte_inbuf_read;
        }

        hyrec.sram_wbuf_read += systolic_array->byte_wbuf_read;
        hyrec.edram_aggr_read += systolic_array->byte_aggrbuf_read;
        hyrec.sram_outbuf_write += systolic_array->byte_outbuf_write;
        hyrec.systolic_add_op += systolic_array->add_op_cnt_;
        hyrec.systolic_mac_op += systolic_array->mac_op_cnt_;

        hyrec.edram_aggr_write += coordinator->byte_aggr_write;
        hyrec.edram_aggr_read += coordinator->byte_aggr_read;
        hyrec.dram_output_write += coordinator->byte_output_write;
        hyrec.dram_input_read += coordinator->byte_input_read;
        hyrec.dram_edge_read += coordinator->byte_edge_read;
        hyrec.dram_weight_read += coordinator->byte_weights_read;

        hyrec.sram_outbuf_read += coordinator->byte_output_write;
        hyrec.sram_input_write += coordinator->byte_input_read;
        hyrec.sram_ebuf_write += coordinator->byte_edge_read;
        hyrec.sram_wbuf_write += coordinator->byte_weights_read;


        dram->PrintStats();
        for(int i = 0; i < dram->ctrls_.size();i++) {
            hyrec.dram_energy += dram->ctrls_[i]->GetStates().calculated_["total_energy"];
            hyrec.dram_dynamic_energy += dram->ctrls_[i]->GetStates().doubles_["act_energy"];
            hyrec.dram_dynamic_energy += dram->ctrls_[i]->GetStates().doubles_["read_energy"];
            hyrec.dram_dynamic_energy += dram->ctrls_[i]->GetStates().doubles_["write_energy"];
            hyrec.dram_dynamic_energy += dram->ctrls_[i]->GetStates().doubles_["ref_energy"];


            hyrec.num_act_cmds += dram->ctrls_[i]->GetStates().counters_["num_act_cmds"];
            hyrec.num_pre_cmds += dram->ctrls_[i]->GetStates().counters_["num_pre_cmds"];
            hyrec.num_read_cmds += dram->ctrls_[i]->GetStates().counters_["num_read_cmds"];
            hyrec.num_write_cmds += dram->ctrls_[i]->GetStates().counters_["num_write_cmds"];
            hyrec.num_ref_cmds += dram->ctrls_[i]->GetStates().counters_["num_ref_cmds"];
            hyrec.num_read_row_hits += dram->ctrls_[i]->GetStates().counters_["num_read_row_hits"];
            hyrec.num_write_row_hits += dram->ctrls_[i]->GetStates().counters_["num_write_row_hits"];
        }
        hyrec.dram_standby_energy = hyrec.dram_energy - hyrec.dram_dynamic_energy;

    }

    std::shared_ptr<HyConfig> hy_config;
    std::shared_ptr<dramsim3::Config> dram_config;
    std::shared_ptr<dramsim3::Config> edram_config;
    std::shared_ptr<dramsim3::JedecDRAMSystem> dram;
    std::shared_ptr<Coordinator> coordinator;
    std::shared_ptr<AggrBuf> aggr_buf;
    std::shared_ptr<SystolicArray> systolic_array;
    HyRec hyrec;

    std::shared_ptr<Aggr> aggr;
    std::shared_ptr<Comb> comb;

    std::deque<EventPtr> dram_return_queue;
    std::deque<EventPtr> edram_return_queue;

    int test = 0;


    bool is_done = false;
    uint64_t clk = 0;

};
#endif //DRAMSIM3_HYGCN_H
