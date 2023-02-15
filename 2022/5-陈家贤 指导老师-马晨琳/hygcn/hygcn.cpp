//
// Created by szu on 2021/11/17.
//

#include "hygcn.h"
#include <functional>
HyGCN::HyGCN(const std::shared_ptr<HyConfig>& hy_config): hy_config(hy_config) {

    dram_config = std::make_shared<dramsim3::Config>("configs/HBM1_8Gb_x128.ini", ".");

    auto callback_func = std::bind(&HyGCN::MemCallback, std::ref(*this), std::placeholders::_1);

    auto buf_callback = std::bind(&HyGCN::BufCallback, std::ref(*this), std::placeholders::_1);


    dram = std::make_shared<dramsim3::JedecDRAMSystem>(*dram_config, ".", callback_func,
                                                       callback_func);

    aggr_buf = std::make_shared<AggrBuf>(hy_config->sys_config->cap_aggrbuf, buf_callback);
    assert(hy_config != nullptr);
    assert(hy_config->sys_config != nullptr);
    int sz_block = hy_config->sys_config->block_size;
    Event::sz_block = sz_block;



    aggr_buf->Init(hy_config->layer_config->space_aligned_vertex(sz_block), sz_block);

    coordinator = std::make_shared<Coordinator>(dram, aggr_buf);

    aggr = std::make_shared<Aggr>(hy_config, aggr_buf, coordinator);

    systolic_array = std::make_shared<SystolicArray>(hy_config->sys_config->systolic_h,
                                                     hy_config->sys_config->systolic_w,
                                                     hy_config->sys_config->cap_outbuf);

    comb = std::make_shared<Comb>(hy_config, aggr_buf, coordinator, systolic_array);
}

void HyGCN::ClockTick() {

    HandleReturnEvent();

    coordinator->ClockTick();

    aggr->ClockTick();
    comb->ClockTick();

    dram->ClockTick();
    aggr_buf->ClockTick();


    clk++;
    if (is_done) {
        std::cout << "clk: " << clk << std::endl;
    }

}

void HyGCN::HandleReturnEvent() {
    while(!dram_return_queue.empty()) {
        auto event = dram_return_queue.front();
        switch(event->data_type) {
            case EventDataType::EDGE:
            case EventDataType::INPUT_FEATURE: {
                aggr->HandleReturnEvent(event);
            }
                break;
            case EventDataType::WEIGHT:
            case EventDataType::OUTPUT_FEATURE: {
                comb->ProcessReturnedEvent(event);
            }
                break;
            default: {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        }
        dram_return_queue.pop_front();
    }


    while(!edram_return_queue.empty())  {
        auto event = edram_return_queue.front();
        switch(event->data_type) {
            case EventDataType::AGGR_FEATURE: {
                if (event->event_type == EventType::WRITE) {
                    aggr->HandleReturnEvent(event);
                } else if(event->event_type == EventType::READ) {
                    comb->ProcessReturnedEvent(event);
                } else if(event->event_type == EventType::READ_AGGR) {
                    aggr->HandleReturnEvent(event);
                }
            }
                break;
            default: {
                dramsim3::AbruptExit(__FILE__, __LINE__);
            }
        }
        edram_return_queue.pop_front();
    }

}
