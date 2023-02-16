//
// Created by szu on 2021/11/17.
//

#include "dataflow/comb.h"


Comb:: Comb(const std::shared_ptr<HyConfig>& hy_config,
            const std::shared_ptr<AggrBuf>& aggr_buf,
            const std::shared_ptr<Coordinator>& coordinator,
            const std::shared_ptr<SystolicArray>& systolic_array):
        hy_config(hy_config),
        systolic_array(systolic_array),
        coordinator(coordinator),
        aggr_buf(aggr_buf),
        sz_max_batch(hy_config->tile_config->sz_comb_batch),
        num_input_chunk_(hy_config->tile_config->num_comb_h_chunk),
        num_output_chunk_(hy_config->tile_config->num_comb_w_chunk),
        sz_input_chunk_(hy_config->tile_config->sz_comb_h_chunk),
        sz_output_chunk_(hy_config->tile_config->sz_comb_w_chunk) {

    input_chunk_cnt_ = 0;
    output_chunk_cnt_ = 0;
    i_slice_cnt_ = 0;
    w_slice_cnt_ = 0;
    o_slice_cnt_ = 0;

    wbuf_ptr_ = -1;
    inbuf_ptr_ = -1;
    filling_outbuf_ptr_ = -1;
    writing_outbuf_ptr_ = -1;

    for (int i = 0; i < 2; i++) {
        inbuf_.emplace_back(hy_config->sys_config->cap_comb_inbuf);
        wbuf_.emplace_back(hy_config->sys_config->cap_wbuf);
    }
    for (int i = 0; i < 2; i++) {
        outbuf_.emplace_back(hy_config->sys_config->cap_outbuf);
    }

    start_vertex = end_vertex = vertex_ptr =  0;
    enable_fetch_ = true;
    halt_vertex = hy_config->loader->raw_graph.num_vertex;
}

void Comb::ClockTick() {


    systolic_array->ClockTick();

    switch(systolic_array->GetState()) {
        case SystolicArray::ArrayState::OUT_SYNC: {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }
            break;
        case SystolicArray::ArrayState::DONE: {
//            std::cout << "array done" << std::endl;
            if(writing_outbuf_ptr_ == -1) { // for next computing
                systolic_array->GetResult();
                // unlock buffer
                if (systolic_array->output_chunk == num_output_chunk_ - 1) {
                    inbuf_[systolic_array->cur_inbuf].SwitchState(SPM::State::IDLE);
                    wbuf_[systolic_array->cur_wbuf].SwitchState(SPM::State::IDLE);
                } else {
                    inbuf_[systolic_array->cur_inbuf].SwitchState(SPM::State::READY);
                    wbuf_[systolic_array->cur_wbuf].SwitchState(SPM::State::IDLE);
                }
                outbuf_[systolic_array->cur_outbuf].SwitchState(SPM::State::WRITING);
                o_slice_cnt_ = 0;
                writing_outbuf_ptr_ = filling_outbuf_ptr_;

            }
        }
            break;
        case SystolicArray::ArrayState::IDLE: {
            if(GetReadyForCompute()) {
                const auto &inbuf = inbuf_[GetTargetBuf(inbuf_, SPM::State::DRAINING)];
                const auto &wbuf = wbuf_[GetTargetBuf(wbuf_, SPM::State::DRAINING)];
                Matrix A(inbuf.matrix_h, inbuf.matrix_w);
                Matrix B(wbuf.matrix_h, wbuf.matrix_w);
                systolic_array->SetMatrix(A, B);

                systolic_array->SetBufConfig(inbuf_ptr_, wbuf_ptr_, filling_outbuf_ptr_);
                systolic_array->SetBlockConfig(input_chunk_cnt_, output_chunk_cnt_);
                FetchNextBlock();
                // std::cout << "array start" << std::endl;
            }
        }
            break;
        default:
            break;
    }

    GenCombDataFlow();

    clk++;

    if(vertex_ptr == halt_vertex){
        if(!is_done) {
            is_done = true;
            end_clk = clk;
        }
    }

}


void Comb::ProcessReturnedEvent(EventPtr event) {

    switch(event->data_type) {
        case EventDataType::AGGR_FEATURE: {
            assert(inbuf_ptr_ != -1);
            inbuf_[inbuf_ptr_].ConfirmEvent();
        }
            break;
        case EventDataType::WEIGHT: {
            assert(wbuf_ptr_ != -1);
            wbuf_[wbuf_ptr_].ConfirmEvent();
        }
            break;
        case EventDataType::OUTPUT_FEATURE: {
            assert(writing_outbuf_ptr_ != -1);
            outbuf_[writing_outbuf_ptr_].ConfirmEvent();
        }
            break;
        default: {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }
            break;
    }

}

void Comb::GenCombDataFlow() {
    if (is_done) {
        return;
    }
    if (vertex_ptr == end_vertex) {
        BatchVertex();
    }

    if(inbuf_ptr_ != -1 && wbuf_ptr_ != -1) {
        if (output_chunk_cnt_ == 0) {
            FetchWeights();
            FetchFeatures();
            if (inbuf_[inbuf_ptr_].EventDone() && wbuf_[wbuf_ptr_].EventDone()) {
                wbuf_[wbuf_ptr_].SwitchState(SPM::State::READY);
                inbuf_[inbuf_ptr_].SwitchState(SPM::State::READY);
            }
        } else {
            FetchWeights();
            if (wbuf_[wbuf_ptr_].EventDone()) {
                wbuf_[wbuf_ptr_].SwitchState(SPM::State::READY);
            }
        }

    } else if(enable_fetch_){
        assert(input_chunk_cnt_ != num_input_chunk_);
        GetReadyForFetch();
    }

    if(writing_outbuf_ptr_ != -1) {
        WriteFeatures();
        if (outbuf_[writing_outbuf_ptr_].EventDone()) {
            outbuf_[writing_outbuf_ptr_].SwitchState(SPM::State::IDLE);
            writing_outbuf_ptr_ = -1;
            if(input_chunk_cnt_ == num_input_chunk_) {
                NextBatch();
            }
        }
    }
}

void Comb::NextBatch() {
    assert(input_chunk_cnt_ == num_input_chunk_);
    input_chunk_cnt_ = 0;
    output_chunk_cnt_ = 0;


    enable_fetch_ = true;
    vertex_ptr = end_vertex;
    aggr_buf->ConfirmComb(end_vertex - start_vertex);

}

void Comb::FetchNextBlock() {
    output_chunk_cnt_++;
    if(output_chunk_cnt_ == num_output_chunk_) {
        output_chunk_cnt_ = 0;
        input_chunk_cnt_++;
        inbuf_ptr_ = -1;
        wbuf_ptr_ = -1;
    } else {
        wbuf_ptr_ = -1;
    }

    if(input_chunk_cnt_ != num_input_chunk_) {
        enable_fetch_ = true;
    }

}


bool Comb::BatchVertex() {

    int ready_num = aggr_buf->GetReadyVertexNum();
    assert(ready_num >= 0);
    if(ready_num > 0) {
        start_vertex = aggr_buf->comb_ptr.first;
        if(ready_num >= sz_max_batch) {
            end_vertex = start_vertex + sz_max_batch;
        } else {
            end_vertex = start_vertex + ready_num;
        }
        vertex_ptr = start_vertex;
        aggr_buf->StartComb(end_vertex - start_vertex);
        return true;
    }

    return false;
}

void Comb::WriteFeatures() {
    const auto &outbuf = outbuf_[writing_outbuf_ptr_];
    if(o_slice_cnt_ >= outbuf.matrix_h) {
        return;
    }

    int sz_block = hy_config->sys_config->block_size;
    int space_vertex = hy_config->layer_config->space_aligned_output_vertex(sz_block);
    uint64_t hex_addr = hy_config->mapper->output_start_addr +
                        (aggr_buf->done_ptr.first + o_slice_cnt_ * space_vertex)  + output_chunk_cnt_ * sz_output_chunk_;
    assert(hex_addr < hy_config->mapper->output_end_addr);

    int num_bytes = outbuf.matrix_w * hy_config->layer_config->data_bytes;
    auto event = std::make_shared<Event>(hex_addr, num_bytes,
                                         EventDirType::DRAM, EventType::WRITE,
                                         EventDataType::OUTPUT_FEATURE);
    coordinator->AddEvent(event);

    o_slice_cnt_++;
}

bool Comb::FetchFeatures() {

    if(i_slice_cnt_ >= end_vertex - start_vertex) {
        return true;
    }

    int sz_block = hy_config->sys_config->block_size;
    int space_vertex = hy_config->layer_config->space_aligned_vertex(sz_block);

    uint64_t hex_addr = aggr_buf->done_ptr.second +
            i_slice_cnt_ * space_vertex + input_chunk_cnt_ * sz_input_chunk_;

    int num_bytes = input_chunk_cnt_ == num_input_chunk_ - 1 ?
                    space_vertex - sz_input_chunk_ * input_chunk_cnt_ : sz_input_chunk_;

    auto event = std::make_shared<Event>(hex_addr, num_bytes,
                                         EventDirType::EDRAM, EventType::READ,
                                         EventDataType::AGGR_FEATURE);
    coordinator->AddEvent(event);

    i_slice_cnt_++;
    return false;
}

bool Comb::FetchWeights() {
    int num_slice = hy_config->tile_config->ele_comb_h_chunk;
    if(w_slice_cnt_ >= num_slice) {
        return true;
    }

    uint64_t hex_addr = hy_config->mapper->weight_start_addr + (input_chunk_cnt_ * num_output_chunk_ + output_chunk_cnt_) *
            hy_config->tile_config->space_weight_block + w_slice_cnt_ * sz_output_chunk_;
    assert(hex_addr < hy_config->mapper->weight_end_addr);
    int num_bytes = hy_config->tile_config->sz_comb_w_chunk;
    auto event = std::make_shared<Event>(hex_addr,
                                         num_bytes, EventDirType::DRAM,
                                         EventType::READ, EventDataType::WEIGHT);
    coordinator->AddEvent(event);

    w_slice_cnt_++;

    return false;
}


void Comb::GetReadyForFetch() {
    if(vertex_ptr == end_vertex) {
        return;
    }

    int sz_batch = end_vertex - start_vertex;

    auto get_inbuf = [&]() {
        int inptr = GetIdleBuf(inbuf_);
        if (inptr == -1) {
            return false;
        }
        inbuf_ptr_ = inptr;
        inbuf_[inptr].SwitchState(SPM::State::FILLING);
        int sz_block = hy_config->sys_config->block_size;
        int space_vertex = hy_config->layer_config->space_aligned_vertex(sz_block);
        int num_bytes = input_chunk_cnt_ == num_input_chunk_ - 1 ?
                        space_vertex - sz_input_chunk_ * input_chunk_cnt_ : sz_input_chunk_;
        inbuf_[inptr].SetMatrixShape((int)sz_batch, num_bytes / hy_config->layer_config->data_bytes);
        inbuf_[inptr].AddEvents((int)sz_batch);
        i_slice_cnt_ = 0;
        return true;
    };

    auto get_wbuf = [&]() {
        int wptr = GetIdleBuf(wbuf_);
        if (wptr == -1) {
            return false;
        }
        wbuf_ptr_ = wptr;
        wbuf_[wptr].SwitchState(SPM::State::FILLING);

        wbuf_[wptr].SetMatrixShape(hy_config->tile_config->ele_comb_h_chunk,
                                   hy_config->tile_config->ele_comb_w_chunk);
        wbuf_[wptr].AddEvents(hy_config->tile_config->ele_comb_h_chunk);
        w_slice_cnt_ = 0;
        return true;
    };

    if(output_chunk_cnt_ == 0) {
        bool res1 = get_inbuf();
        bool res2 = get_wbuf();
        assert(res1 && res2);
    } else {
        bool res = get_wbuf();
        assert(res);
    }
    enable_fetch_ = false;
}

bool Comb::GetReadyForCompute() {
    int inptr = GetReadyBuf(inbuf_);
    int wptr = GetReadyBuf(wbuf_);
    int outptr = GetIdleBuf(outbuf_);
    int res = inptr != -1 && wptr != -1 && outptr != -1;
    if(res) {
        inbuf_[inptr].SwitchState(SPM::State::DRAINING);
        wbuf_[wptr].SwitchState(SPM::State::DRAINING);
        outbuf_[outptr].SwitchState(SPM::State::FILLING);
        filling_outbuf_ptr_ = outptr;
        outbuf_[outptr].AddEvents(inbuf_[inptr].matrix_h);
        outbuf_[outptr].SetMatrixShape(inbuf_[inptr].matrix_h, wbuf_[wptr].matrix_w);
    }
    return res;
}


int Comb::GetIdleBuf(const std::vector<SPM> &buf) {
    return GetTargetBuf(buf, SPM::State::IDLE);
}

int Comb::GetReadyBuf(const std::vector<SPM> &buf) {
    return GetTargetBuf(buf, SPM::State::READY);
}

int Comb::GetTargetBuf(const std::vector<SPM> &buf, SPM::State state) {
    for (int i = 0; i < buf.size(); i++) {
        if(buf[i].GetState() == state) {
            return i;
        }
    }
    return -1;
}


