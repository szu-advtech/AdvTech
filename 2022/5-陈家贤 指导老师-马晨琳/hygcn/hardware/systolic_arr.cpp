//
// Created by szu on 2021/11/19.
//


#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "hardware/systolic_array.h"

SystolicArray::SystolicArray(int height, int width, int cap_outbuf, int data_bytes):
        height_(height),
        width_(width),
        cap_outbuf_(cap_outbuf),
        data_bytes_(data_bytes){
    state = ArrayState::IDLE;
    fill_weight_cycle = compute_cycle = 0;
    mac_op_cnt_ = add_op_cnt_ = 0;
    computing_cycle_ = 0;
    filling_cycle_ = 0;
}

void SystolicArray::FillWeight(const Matrix &weights) {
    // assert(weights.w == width_ && weights.h == height_); padding with zero

    state = ArrayState::FILLING;
    // todo is it right?
    // fill_weight_cycle = std::min(width_, height_);
    fill_weight_cycle = 0;
    byte_wbuf_read += weights.h * weights.w;
}

void SystolicArray::FillInputFIFO(const Matrix &inputs) {
    // assert(inputs.w == height_); padding with zero

    // compute_cycle = inputs.h + height_ + width_ - 2;
    // another one cycle for accumulation
    compute_cycle = inputs.h + height_ + width_ - 1;
    // mac_op_cnt_ += compute_cycle * height_ * width_;
    mac_op_cnt_ += compute_cycle * height_ * width_;
    add_op_cnt_ += inputs.h * width_; // todo add redundancy

    byte_aggrbuf_read += inputs.h * inputs.w + inputs.h * width_;
    byte_outbuf_write += inputs.h * width_;
}

SystolicArray::ArrayState SystolicArray::GetState() {
    return state;
}

bool SystolicArray::IsIdle() {
    return state == ArrayState::IDLE;
}

void SystolicArray::GenDataFlow() {

    if(h_tile_cnt_ == 0) {
        if (sz_output_ + W.w * data_bytes_ > cap_outbuf_) {
            state = ArrayState::OUT_SYNC;
            return;
        }
        sz_output_ += W.w * data_bytes_;
    }

    int batch = batch_cnt_ == num_batch_ - 1 ? ele_last_batch_ : ele_batch_;
    FillWeight(Matrix(ele_h_tile_, ele_w_tile_));
    FillInputFIFO(Matrix(batch, ele_h_tile_));
    h_tile_cnt_++;

    if(h_tile_cnt_ == num_h_tile_) {
        w_tile_cnt_++;
        h_tile_cnt_ = 0;
        if(w_tile_cnt_ == num_w_tile_) {
            batch_cnt_++;
            w_tile_cnt_ = 0;
            if(batch_cnt_ == num_batch_) {
                state = ArrayState::DONE;
            }
        }
    }

}

void SystolicArray::ClockTick() {
    switch(state) {
        case ArrayState::FILLING: {
            fill_weight_cycle--;
            if(fill_weight_cycle <= 0) {
                state = ArrayState::COMPUTING;
            }
        }
            filling_cycle_++;
            break;
        case ArrayState::COMPUTING: {
            compute_cycle--;
            if(compute_cycle == 0) {
                state = ArrayState::READY;
            }
        }
            // std::cout << "computing" << std::endl;
            computing_cycle_++;
            break;
        case ArrayState::READY: {
            GenDataFlow();
        }
            break;
        case ArrayState::IDLE:
        case ArrayState::OUT_SYNC:
        case ArrayState::DONE:
        default:
            break;
    }
}

void SystolicArray::SetMatrix(const Matrix& a, const Matrix &w) {
    A = a;
    W = w;
    ele_h_tile_ = height_;
    ele_w_tile_ = width_;
    ele_batch_ = std::min(cap_outbuf_ / (w.w * data_bytes_), a.h);

    num_h_tile_ = std::ceil((double)w.h / ele_h_tile_);
    num_w_tile_ = std::ceil((double)w.w / ele_w_tile_);
    num_batch_ = std::ceil((double)a.h / ele_batch_);
    ele_last_batch_ = a.h - ele_batch_ * (num_batch_ - 1);
    h_tile_cnt_ = 0;
    w_tile_cnt_ = 0;
    batch_cnt_ = 0;
    sz_output_ = 0;
    state = ArrayState::READY;
    assert(cap_outbuf_ >= A.h * ele_w_tile_);
}

int SystolicArray::GetNumOutputVertex() {
    return sz_output_ / (W.w * data_bytes_);
}

void SystolicArray::SyncOutput() {
    state = ArrayState::READY;
    sz_output_ = 0;
}

void SystolicArray::GetResult() {
    state = ArrayState::IDLE;
}

void SystolicArray::SetBufConfig(int inbuf, int wbuf, int outbuf) {
    cur_inbuf = inbuf;
    cur_wbuf = wbuf;
    cur_outbuf = outbuf;
}

void SystolicArray::SetBlockConfig(int ichunk, int ochunk) {
    input_chunk = ichunk;
    output_chunk = ochunk;
}