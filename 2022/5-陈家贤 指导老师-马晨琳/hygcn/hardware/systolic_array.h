//
// Created by szu on 2021/11/19.
//

#ifndef DRAMSIM3_SYSTOLIC_ARRAY_H
#define DRAMSIM3_SYSTOLIC_ARRAY_H
#include <cstdint>

class Matrix {
public:
    Matrix() {}
    Matrix(int h, int w): h(h), w(w) {}
    int h;
    int w;
};

class SystolicArray {
public:

    enum class ArrayState {
        FILLING,
        COMPUTING,
        READY,
        OUT_SYNC,
        DONE,
        IDLE
    };

    SystolicArray(int height, int width, int cap_outbuf, int data_bytes = 4);
    void ClockTick();
    bool IsIdle();
    bool IsReady();

    int GetNumOutputVertex();

    void SyncOutput();

    void GetResult();

    ArrayState GetState();

    void SetMatrix(const Matrix& a, const Matrix &w);

    void SetBufConfig(int inbuf, int wbuf, int outbuf);

    void SetBlockConfig(int ichunk, int ochunk);

    void GenDataFlow();


    int cur_inbuf;
    int cur_outbuf;
    int cur_wbuf;

    int input_chunk;
    int output_chunk;

    uint64_t computing_cycle_;
    uint64_t filling_cycle_;
    uint64_t mac_op_cnt_;
    uint64_t add_op_cnt_;
    uint64_t byte_wbuf_read = 0;
    uint64_t byte_aggrbuf_read = 0;
    uint64_t byte_outbuf_write = 0;

private:

    void FillWeight(const Matrix &weights);
    void FillInputFIFO(const Matrix &inputs);


    Matrix A, W;
    // constant
    const int height_;
    const int width_;
    const int cap_outbuf_;
    const int data_bytes_;

    ArrayState state;

    int fill_weight_cycle;
    int compute_cycle;

    int sz_output_;

    int ele_h_tile_;
    int ele_w_tile_;
    int ele_batch_;
    int ele_last_batch_;
    int num_h_tile_;
    int num_w_tile_;
    int num_batch_;
    int h_tile_cnt_;
    int w_tile_cnt_;
    int batch_cnt_;

};
#endif //DRAMSIM3_SYSTOLIC_ARRAY_H
