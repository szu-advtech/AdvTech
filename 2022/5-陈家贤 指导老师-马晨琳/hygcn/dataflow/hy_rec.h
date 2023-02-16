//
// Created by szu on 2021/11/20.
//

#ifndef DRAMSIM3_HY_REC_H
#define DRAMSIM3_HY_REC_H

#include <string>
#include <sstream>

class HyRec {
public:
    std::string GetString() const {

        std::stringstream output;

        output << finish_time << ","
                << aggr_end_time << ","
                << comb_end_time << ","
                << dram_energy << ","
                << dram_dynamic_energy << ","
                << dram_standby_energy << ","
                << dram_edge_read << ","
                << dram_input_read << ","
                << edram_aggr_read << ","
                << edram_aggr_write << ","
                << dram_weight_read << ","
                << dram_output_write << ","
                << sram_ebuf_read << ","
                << sram_ebuf_write << ","
                << sram_inbuf_read << ","
                << sram_input_write << ","
                << sram_wbuf_read << ","
                << sram_wbuf_write << ","
                << sram_outbuf_read << ","
                << sram_outbuf_write << ","
                << simd_mac_op << ","
                << systolic_mac_op << ","
                << systolic_add_op << ","
                << num_act_cmds << ","
                << num_pre_cmds << ","
                << num_read_cmds << ","
                << num_write_cmds << ","
                << num_ref_cmds << ","
                << num_read_row_hits << ","
                << num_write_row_hits << '\n';

        return output.str();
    }

    static std::string GetHeaderString()  {

        std::stringstream output;

        output << "finish_time" << ","
               << "aggr_end_time" << ","
               << "comb_end_time" << ","
               << "dram_energy" << ","
               << "dram_dynamic_energy" << ","
               << "dram_standby_energy" << ","
               << "dram_edge_read" << ","
               << "dram_input_read" << ","
               << "edram_aggr_read" << ","
               << "edram_aggr_write" << ","
               << "dram_weight_read" << ","
               << "dram_output_write" << ","
               << "sram_ebuf_read" << ","
               << "sram_ebuf_write" << ","
               << "sram_inbuf_read" << ","
               << "sram_input_write" << ","
               << "sram_wbuf_read" << ","
               << "sram_wbuf_write" << ","
               << "sram_outbuf_read" << ","
               << "sram_outbuf_write" << ","
               << "simd_mac_op" << ","
               << "systolic_mac_op" << ","
               << "systolic_add_op" << ","
               << "num_act_cmds" << ","
               << "num_pre_cmds" << ","
               << "num_read_cmds" << ","
               << "num_write_cmds" << ","
               << "num_ref_cmds" << ","
               << "num_read_row_hits" << ","
               << "num_write_row_hits" << '\n';

        return output.str();
    }

    uint64_t finish_time = 0;
    uint64_t aggr_end_time = 0;
    uint64_t comb_end_time = 0;

    double dram_energy = 0; //pj
    double dram_dynamic_energy = 0;
    double dram_standby_energy = 0;

    uint64_t dram_edge_read = 0;
    uint64_t dram_input_read = 0;
    uint64_t edram_aggr_read = 0;
    uint64_t edram_aggr_write = 0;
    uint64_t dram_weight_read = 0;
    uint64_t dram_output_write = 0;

    uint64_t sram_ebuf_read = 0;
    uint64_t sram_ebuf_write = 0;
    uint64_t sram_inbuf_read = 0;
    uint64_t sram_input_write = 0;
    uint64_t sram_wbuf_read = 0;
    uint64_t sram_wbuf_write = 0;
    uint64_t sram_outbuf_read = 0;
    uint64_t sram_outbuf_write = 0;

    uint64_t simd_mac_op = 0;
    uint64_t systolic_mac_op = 0;
    uint64_t systolic_add_op = 0;


    uint64_t num_act_cmds = 0;
    uint64_t num_pre_cmds = 0;
    uint64_t num_read_cmds = 0;
    uint64_t num_write_cmds = 0;
    uint64_t num_ref_cmds = 0;
    uint64_t num_read_row_hits = 0;
    uint64_t num_write_row_hits = 0;


};

#endif //DRAMSIM3_HY_REC_H
