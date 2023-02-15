//
// Created by szu on 2021/11/17.
//
#include <deque>
#include "aggr.h"
#include "config.h"
#include "event.h"

Aggr::Aggr(const std::shared_ptr<HyConfig>& hy_config,
           const std::shared_ptr<AggrBuf>& aggr_buf,
           const std::shared_ptr<Coordinator>& coordinator):
    hy_config(hy_config),
    aggr_buf(aggr_buf),
    coordinator(coordinator),
    edge_chunk_cnt(0),
    is_done(false),
    num_edge_chunk(hy_config->mapper->num_edge_chunk) {
    for(int i = 0; i < 2; i++) {
        inbuf.emplace_back(std::make_shared<SPMA>(hy_config->sys_config->cap_inbuf));
        ebuf.emplace_back(std::make_shared<SPMA>(hy_config->sys_config->cap_edgebuf));
    }

    int num_simd = hy_config->sys_config->num_simd;
    int num_core_per_simd = hy_config->sys_config->num_core_per_simd;
    simds.reserve(num_simd);
    for(int i = 0; i < num_simd; i++) {
        simds.emplace_back(std::make_shared<SIMD>(num_core_per_simd));
    }

}

void Aggr::ClockTick() {
    for(auto &simd : simds) {
        simd->ClockTick();
        if(simd->IsReady()) {
            assert(inbuf[simd->inbuf_ptr]->GetState() == SPMA::State::DRAINING);
            uint64_t addr = aggr_buf->GetAggrWriteAddr(simd->vertex_id, simd->feature_chunk_id);
            int bytes = simd->writeback_byte;
            EventPtr event = std::make_shared<Event>(addr, bytes, EventDirType::EDRAM,
                                                     EventType::WRITE, EventDataType::AGGR_FEATURE);
            coordinator->AddEvent(event);
            event_out++;

            auto ibuf = inbuf[simd->inbuf_ptr];
            ibuf->ConfirmEvent();
            if(ibuf->EventDone()) {
                int bind_ebuf = ibuf->ebuf_ptr;
                ebuf[bind_ebuf]->ConfirmEvent();
                if(ebuf[bind_ebuf]->EventDone()) {
                    assert(ebuf[bind_ebuf]->GetState() == SPMA::State::DRAINING);
                    int num_vertex = ebuf[bind_ebuf]->end_vertex - ebuf[bind_ebuf]->start_vertex;
                    aggr_buf->ConfirmAggr(num_vertex);
                    ebuf[bind_ebuf]->SwitchState(SPMA::State::IDLE);
                }
                ibuf->SwitchState(SPMA::State::IDLE);
            }
            simd->SwitchState(SIMD::State::IDLE);
        }
    }


    GenDataFlow();
    clk++;
    if(AllBufIdle(inbuf) && AllBufIdle(ebuf) && event_out == event_in) {
        if(!is_done) {
            is_done = true;
            end_clk = clk;
        }

    }

}

void Aggr::GenDataFlow() {
    if(is_done) {
        return;
    }

    // simd
    int width = hy_config->sys_config->num_core_per_simd;
    int num_feature_chunk = std::ceil((double) hy_config->layer_config->weight_h / width);

    // edge buffer
    if(edge_chunk_cnt != num_edge_chunk) {
        int idle_ebuf = GetTargetBuf(ebuf, SPMA::State::IDLE);
        if(idle_ebuf != -1 && GetTargetBuf(ebuf, SPMA::State::FILLING) == -1) {
            ebuf[idle_ebuf]->SwitchState(SPMA::State::FILLING);
            ebuf[idle_ebuf]->AddEvents(1);
            int bytes = std::min(hy_config->sys_config->cap_edgebuf, hy_config->tile_config->space_aligned_weight);
            uint64_t addr = hy_config->mapper->edge_start_addr + edge_chunk_cnt * bytes;
            assert(addr < hy_config->mapper->edge_end_addr);
            EventPtr event = std::make_shared<Event>(addr, bytes, EventDirType::DRAM,
                                                     EventType::READ, EventDataType::EDGE);
            coordinator->AddEvent(event);
            {
                ebuf[idle_ebuf]->edge_chunk_id = edge_chunk_cnt;
                ebuf[idle_ebuf]->start_vertex = hy_config->mapper->start_vertex_of_echunk[edge_chunk_cnt];
                ebuf[idle_ebuf]->end_vertex = hy_config->mapper->end_vertex_of_echunk[edge_chunk_cnt];
                auto edge_chunk = hy_config->mapper->edge_chunk[edge_chunk_cnt];
                ebuf[idle_ebuf]->edge_parts = Partition(edge_chunk);
                ebuf[idle_ebuf]->num_edge_part = ebuf[idle_ebuf]->edge_parts.size();
            }
            edge_chunk_cnt++;
        }
    }

    // input buf
    int idle_input_ptr = GetTargetBuf(inbuf, SPMA::State::IDLE);
    int ready_ebuf_ptr = GetTargetBuf(ebuf, SPMA::State::READY);
    int draining_ebuf_ptr = GetTargetBuf(ebuf, SPMA::State::DRAINING);

    if(idle_input_ptr != -1 && GetTargetBuf(inbuf, SPMA::State::FILLING) == -1) {


        int bind_ebuf_ptr = -1;
        int edge_part_id = -1;

        if (draining_ebuf_ptr != -1) {
            edge_part_id = ebuf[draining_ebuf_ptr]->edge_part_cnt;
            if(edge_part_id < ebuf[draining_ebuf_ptr]->num_edge_part) {
                ebuf[draining_ebuf_ptr]->edge_part_cnt++;
                bind_ebuf_ptr = draining_ebuf_ptr;
            }

        } else if(ready_ebuf_ptr != -1) {

            // magaic, select ebuf with min start vertex
            int ready_ebuf_ptr = -1;
            int ebuf_start_vertex = INT32_MAX;
            for(int i = 0; i < ebuf.size(); i++) {
                const auto& buf = ebuf[i];
                if(buf->GetState() == SPMA::State::READY) {
                    if(ebuf_start_vertex > buf->start_vertex) {
                        ebuf_start_vertex = buf->start_vertex;
                        ready_ebuf_ptr = i;
                    }
                }
            }

            int num_vertex = ebuf[ready_ebuf_ptr]->end_vertex -
                             ebuf[ready_ebuf_ptr]->start_vertex;
            if(aggr_buf->WillAllocate(num_vertex)) {
                edge_part_id = ebuf[ready_ebuf_ptr]->edge_part_cnt;
                assert(edge_part_id < ebuf[ready_ebuf_ptr]->num_edge_part);
                ebuf[ready_ebuf_ptr]->SwitchState(SPMA::State::DRAINING);
                ebuf[ready_ebuf_ptr]->AddEvents(ebuf[ready_ebuf_ptr]->num_edge_part);
                ebuf[ready_ebuf_ptr]->edge_part_cnt++;
                bind_ebuf_ptr = ready_ebuf_ptr;
                aggr_buf->StartAggr(num_vertex);
            }
        }


        if(bind_ebuf_ptr != -1) {

            inbuf[idle_input_ptr]->SwitchState(SPMA::State::FILLING);
            inbuf[idle_input_ptr]->start_vertex = ebuf[bind_ebuf_ptr]->start_vertex;
            inbuf[idle_input_ptr]->end_vertex = ebuf[bind_ebuf_ptr]->end_vertex;
            inbuf[idle_input_ptr]->input_edge = ebuf[bind_ebuf_ptr]->edge_parts[edge_part_id];
            for(int i = 0; i < inbuf[idle_input_ptr]->input_edge.size(); i++) {
                if(!inbuf[idle_input_ptr]->input_edge[i].empty()) {
                    inbuf[idle_input_ptr]->end_vertex = i + inbuf[idle_input_ptr]->start_vertex + 1;
                    inbuf[idle_input_ptr]->non_empty_vertex += 1;
                }
            }
            inbuf[idle_input_ptr]->neighbors = SparsityElimination(inbuf[idle_input_ptr]->input_edge);
            inbuf[idle_input_ptr]->ebuf_ptr = bind_ebuf_ptr;
            inbuf[idle_input_ptr]->num_feature_chunk = num_feature_chunk;

            for(auto vertex_id : inbuf[idle_input_ptr]->neighbors) {
                int sz_block = hy_config->sys_config->block_size;
                int bytes = hy_config->layer_config->space_aligned_vertex(sz_block);
                uint64_t addr = hy_config->mapper->input_start_addr +
                                vertex_id * bytes;
                assert(addr < hy_config->mapper->input_end_addr);
                EventPtr event = std::make_shared<Event>(addr, bytes, EventDirType::DRAM,
                                                         EventType::READ, EventDataType::INPUT_FEATURE);
                coordinator->AddEvent(event);
            }
            inbuf[idle_input_ptr]->AddEvents(inbuf[idle_input_ptr]->neighbors.size());
        }

    }

    // simd

    int ready_inbuf_ptr = GetTargetBuf(inbuf, SPMA::State::READY);
    int draining_inbuf_ptr = GetTargetBuf(inbuf, SPMA::State::DRAINING);
    if(ready_inbuf_ptr != -1 || draining_inbuf_ptr != -1) {
        std::shared_ptr<SPMA> buf;
        int ptr = -1;
        if(draining_inbuf_ptr != -1) {
            buf = inbuf[draining_inbuf_ptr];
            ptr = draining_inbuf_ptr;
        } else if(ready_inbuf_ptr != -1) {
            buf = inbuf[ready_inbuf_ptr];
            ptr = ready_inbuf_ptr;
            inbuf[ready_inbuf_ptr]->AddEvents(inbuf[ready_inbuf_ptr]->non_empty_vertex * num_feature_chunk);
//            inbuf[ready_inbuf_ptr]->AddEvents(inbuf[ready_inbuf_ptr]->non_empty_vertex * num_feature_chunk * 2);
//            nmsl = inbuf[ready_inbuf_ptr]->non_empty_vertex * num_feature_chunk * 2;

            inbuf[ready_inbuf_ptr]->SwitchState(SPMA::State::DRAINING);
        }

        if(ptr != -1) {
            while(true) {
                int idle_simd = GetIdleSIMD();
                if(idle_simd == -1) {
                    break;
                }
                if(buf->vertex_ptr < buf->end_vertex) {
                    int depth = buf->input_edge[buf->vertex_ptr - buf->start_vertex].size();
                    if(depth == 0) {
                        buf->vertex_ptr++;
                        continue;
                    }
                    simds[idle_simd]->vertex_id = buf->vertex_ptr;
                    simds[idle_simd]->inbuf_ptr = ptr;
                    simds[idle_simd]->feature_chunk_id = buf->feature_chunk_cnt;
                    simds[idle_simd]->AssignTask(width, depth);

                    // some magic again, "+1", maybe hash conflict , for dram not sram
                    uint64_t addr = aggr_buf->GetAggrWriteAddr(simds[idle_simd]->vertex_id, simds[idle_simd]->feature_chunk_id);
                    int bytes = simds[idle_simd]->writeback_byte;
                    EventPtr event = std::make_shared<Event>(addr, bytes, EventDirType::EDRAM,
                                                             EventType::READ_AGGR, EventDataType::AGGR_FEATURE);
                    coordinator->AddEvent(event);
                    event_out++;

                    buf->feature_chunk_cnt++;
                    if(buf->feature_chunk_cnt == num_feature_chunk) {
                        buf->feature_chunk_cnt = 0;
                        buf->vertex_ptr++;
                    }
                } else {
                    break;
                }
            }
        }

    }




}

void Aggr::HandleReturnEvent(EventPtr event) {
    switch(event->data_type) {
        case EventDataType::EDGE: {
            int filling_ebuf = GetTargetBuf(ebuf, SPMA::State::FILLING);
            assert(filling_ebuf != -1);
            ebuf[filling_ebuf]->ConfirmEvent();
            if(ebuf[filling_ebuf]->EventDone()) {
                ebuf[filling_ebuf]->SwitchState(SPMA::State::READY);
            }

        }
            break;
        case EventDataType::INPUT_FEATURE: {
            int filling_inbuf = GetTargetBuf(inbuf, SPMA::State::FILLING);
            assert(filling_inbuf != -1);
            inbuf[filling_inbuf]->ConfirmEvent();
            if(inbuf[filling_inbuf]->EventDone()) {
                inbuf[filling_inbuf]->SwitchState(SPMA::State::READY);
//                inbuf[filling_inbuf]->SwitchState(SPMA::State::DRAINING);
//                inbuf[filling_inbuf]->SwitchState(SPMA::State::IDLE);
            }
        }
            break;
        case EventDataType::AGGR_FEATURE: {
            event_in++;
        }
            break;
        default: {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }
    }

}

std::vector<std::vector<std::vector<int>>> Aggr::Partition(const std::vector<std::vector<int>>& edges) {
    int num_aggr_vertex = edges.size();
    std::vector<std::vector<std::vector<int>>> partitions;
    std::vector<std::deque<int>> edge_set;
    edge_set.reserve(num_aggr_vertex);
    for(int i = 0; i < num_aggr_vertex; i++) {
        edge_set.emplace_back(edges[i].begin(), edges[i].end());
    }

    int ele_h_interval = hy_config->tile_config->ele_aggr_h_interval;

    int end_flag = false;

    while(!end_flag) {
        int start_edge = hy_config->loader->raw_graph.num_vertex;
        int end_edge = start_edge + ele_h_interval;
        std::vector<int> stop_point(num_aggr_vertex, -1);
        std::vector<bool> stop_flag(num_aggr_vertex, false);
        for(int i = 0; i < num_aggr_vertex; i++) {
            if(!edge_set[i].empty()) {
                start_edge =  std::min(start_edge, edge_set[i][0]);
            }
        }
        end_edge = start_edge + ele_h_interval;

        for(int i = 0; i < num_aggr_vertex; i++) {
            if(stop_flag[i]) {
                continue;
            }
            for(int j = 0; j <= edge_set[i].size() && j <= ele_h_interval; j++) {
                if(j == edge_set[i].size() || j == ele_h_interval) {
                    stop_point[i] = j;
                    stop_flag[i] = true;
                    break;
                }
                if(edge_set[i][j] >= end_edge) {
                    stop_point[i] = j;
                    stop_flag[i] = true;
                    break;
                }
            }
        }
        std::vector<std::vector<int>> part;
        part.reserve(num_aggr_vertex);
        for(int i = 0; i < num_aggr_vertex; i++) {
            assert(stop_point[i] != -1 && stop_point[i] <= edge_set[i].size());
            assert(stop_flag[i] == true);
            part.emplace_back(edge_set[i].begin(), edge_set[i].begin() + stop_point[i]);
            edge_set[i].erase(edge_set[i].begin(), edge_set[i].begin() + stop_point[i]);
        }
        partitions.emplace_back(part);

        end_flag = true;
        for(int i = 0; i < num_aggr_vertex; i++) {
            if(!edge_set[i].empty()) {
                end_flag = false;
            }
        }

    }
    return partitions;
}

std::set<int> Aggr::SparsityElimination(std::vector<std::vector<int>> edges) {
    std::set<int> set;
    for(int i = 0; i < edges.size(); i++) {
        for(int j = 0; j < edges[i].size(); j++) {
            set.insert(edges[i][j]);
        }
    }
    return set;
}

int Aggr::GetIdleSIMD() {
    for(int i = 0; i < simds.size(); i++) {
        if(simds[i]->IsIdle()) {
            return i;
        }
    }
    return -1;
}

