//
// Created by szu on 2021/11/17.
//

#include "hardware/coordinator.h"

void Coordinator::AddEvent(EventPtr event) {

    if(event->dir_type == EventDirType::DRAM) {
        switch(event->data_type) {
            case EventDataType::EDGE: {
                edges.emplace_back(event);
                byte_edge_read += event->bytes;
            }
                break;
            case EventDataType::WEIGHT: {
                weights.emplace_back(event);
                byte_weights_read += event->bytes;
            }
                break;
            case EventDataType::INPUT_FEATURE: {
                inputs.emplace_back(event);
                byte_input_read += event->bytes;
            }
                break;
            case EventDataType::OUTPUT_FEATURE: {
                outputs.emplace_back(event);
                byte_output_write += event->bytes;
            }
                break;
            default: {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        }
    } else if(event->dir_type == EventDirType::EDRAM) {
        if(event->data_type == EventDataType::AGGR_FEATURE) {
            tmps.emplace_back(event);
            if(event->event_type == EventType::WRITE) {
                byte_aggr_write += event->bytes;
            } else if(event->event_type == EventType::READ) {
                byte_aggr_read += event->bytes;
            } else if(event->event_type == EventType::READ_AGGR) {
                byte_aggr_read += event->bytes;
            } else {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        } else {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }
    } else {
        dramsim3::AbruptExit(__FILE__, __LINE__);
    }
        return;
}

void Coordinator::IssueEvent() {
    std::deque<EventPtr> *queue = nullptr;

    if(!edges.empty()) {
        queue = &edges;
    } else if(!inputs.empty()) {
        queue = &inputs;
    } else if(!weights.empty()) {
        queue = &weights;
    } else if(!outputs.empty()) {
        queue = &outputs;
    }

    if(queue != nullptr && !queue->empty()) {
        auto event = queue->front();
        bool is_write = event->event_type == EventType::WRITE;
        while(!event->AllTransIssued()) {
            uint64_t addr = event->address + Event::sz_block * event->trans_cnt;

            if(dram->WillAcceptTransaction(addr, is_write)) {
                events.insert(std::make_pair(addr, event));
                dram->AddTransaction(addr, is_write);
                event->IssueTrans();

            } else {
                break;
            }
        }

        if(event->AllTransIssued()) {
            queue->pop_front();
        }
    }


    queue = nullptr;
    if(!tmps.empty()) {
        queue = &tmps;
    }
    while(queue != nullptr && !queue->empty()) {
        auto event = queue->front();
        bool is_write = event->event_type == EventType::WRITE;
        while (!event->AllTransIssued()) {
            uint64_t addr = event->address + Event::sz_block * event->trans_cnt;
            if(addr >= edram->cap_edram) {
                addr = (addr + edram->cap_edram) % (edram->cap_edram);
            }
            if(addr >= edram->cap_edram && addr + Event::sz_block > edram->cap_edram) {
                assert(addr < edram->cap_edram && addr + Event::sz_block <= edram->cap_edram);
            }


            if (edram->WillAcceptTransaction(addr, is_write)) {
                edram_events.insert(std::make_pair(addr, event));
                edram->AddTransaction(addr, is_write);
                event->IssueTrans();

            } else {
                break;
            }
        }

        if (event->AllTransIssued()) {
            queue->pop_front();
        }

        if(!edram->WillAcceptTransaction(0, false)) {
            break;
        }
    }
}

void Coordinator::ClockTick() {
    IssueEvent();
}
