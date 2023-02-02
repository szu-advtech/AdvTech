//
// Created by szu on 2021/11/17.
//

#ifndef DRAMSIM3_COMB_H
#define DRAMSIM3_COMB_H

#include <memory>
#include <deque>
#include "hardware/coordinator.h"
#include "hardware/systolic_array.h"
#include "hardware/spm.h"
#include "config.h"
#include "event.h"

class Comb {
public:
    Comb(const std::shared_ptr<HyConfig>& hy_config,
         const std::shared_ptr<AggrBuf>& aggr_buf,
         const std::shared_ptr<Coordinator>& coordinator,
         const std::shared_ptr<SystolicArray>& systolic_array);

    void ClockTick();

    void ProcessReturnedEvent(EventPtr event);

    void SetHaltVertex(int vid) {
        halt_vertex = vid;
    }

    std::shared_ptr<HyConfig> hy_config;
    std::shared_ptr<Coordinator> coordinator;
    std::shared_ptr<AggrBuf> aggr_buf;
    std::shared_ptr<SystolicArray> systolic_array;

    uint64_t clk = 0;
    uint64_t end_clk = 0;
    bool is_done = false;

private:
    bool BatchVertex();

    void GenCombDataFlow();

    bool FetchFeatures();

    bool FetchWeights();

    void WriteFeatures();

    void NextBatch();

    void GetReadyForFetch();

    bool GetReadyForCompute();

    void FetchNextBlock();

    static int GetIdleBuf(const std::vector<SPM> &buf);

    static int GetReadyBuf(const std::vector<SPM> &buf);

    static int GetTargetBuf(const std::vector<SPM> &buf, SPM::State state);




    // constant

    const int num_input_chunk_;
    const int num_output_chunk_;
    const int sz_input_chunk_;
    const int sz_output_chunk_;

    const int sz_max_batch;

    int input_chunk_cnt_;
    int output_chunk_cnt_;
    int w_slice_cnt_;
    int i_slice_cnt_;
    int o_slice_cnt_;

    int inbuf_ptr_; // filling
    int wbuf_ptr_; // filling
    int filling_outbuf_ptr_; // filling
    int writing_outbuf_ptr_; // writing

    bool enable_fetch_;

    //  components double buffer
    std::vector<SPM> inbuf_;
    std::vector<SPM> wbuf_;
    std::vector<SPM> outbuf_;


    int start_vertex = 0;
    int end_vertex = 0;
    int vertex_ptr;

    int halt_vertex = -1;

};
#endif //DRAMSIM3_COMB_H
