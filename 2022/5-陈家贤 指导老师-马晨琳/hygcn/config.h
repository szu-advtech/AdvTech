//
// Created by szu on 2021/11/15.
//

#ifndef DRAMSIM3_CONFIG_H
#define DRAMSIM3_CONFIG_H

#include "INIReader.h"
#include <cassert>
#include <iostream>
#include <memory>

#include "common.h"
#include "graph.h"
#include "tool.h"

class SysConfig;
class LayerConfig;
class AddrMapper;
class TileConfig;

class HyConfig {
public:
    HyConfig() {
    }
    void Init(const std::string& graph_path, const std::string& graph_name, const std::string& sys_path, const std::string& layer_path);
    std::shared_ptr<SysConfig> sys_config;
    std::shared_ptr<LayerConfig> layer_config;
    std::shared_ptr<GraphLoader> loader;
    std::shared_ptr<AddrMapper> mapper;
    std::shared_ptr<TileConfig> tile_config;
    bool PrepareNextLayer();
};

class SysConfig {
public:
    SysConfig(const std::string& config_path);

    int block_size;
    uint64_t cap_hbm;

    int num_simd;
    int num_core_per_simd;
    int systolic_h;
    int systolic_w;

    int cap_inbuf;
    int cap_edgebuf;
    int cap_wbuf;
    int cap_outbuf;
    int cap_aggrbuf;

    int cap_comb_inbuf;
    int cap_comb_outbuf;
};


class LayerConfig {
   public:
    LayerConfig(const std::string& layer_path);

    int len_features() const {
        return weight_h;
    }

    int len_output_features() const {
        return weight_w;
    }

    int sz_vertex() const {
        return data_bytes * len_features();
    }

    int space_aligned_vertex(int sz_block) const {
        return UpperAlign(data_bytes * len_features(), sz_block);
    }

    int sz_output_vertex() const {
        return data_bytes * len_output_features();
    }

    int space_aligned_output_vertex(int sz_block) const {
        return UpperAlign(data_bytes * len_output_features(), sz_block);
    }

    std::shared_ptr<INIReader> layer_reader;
    std::string model_name;
    int sample = -1;
    int weight_h; // len_features
    int weight_w; // len_output_features
    int data_bytes;
    int idx_bytes;
    int num_layer;
    int layer_count;
};

class TileConfig {
public:
    TileConfig(const HyConfig& hy_config);


    int sz_comb_batch;

    int sz_comb_h_chunk;
    int sz_comb_w_chunk;

    int ele_comb_h_chunk;
    int ele_comb_w_chunk;

    int num_comb_h_chunk;
    int num_comb_w_chunk;

    int aligned_weight_h;
    int aligned_weight_w;

    int space_weight_block;

    int space_aligned_weight;

    int ele_aggr_h_interval;
    int ele_aggr_w_interval;

};

class AddrMapper {
public:
    AddrMapper (const std::shared_ptr<GraphLoader> &loader,
                const std::shared_ptr<SysConfig> &sys_config,
                const std::shared_ptr<LayerConfig> &layer_config,
                const std::shared_ptr<TileConfig> &tile_config);


    const int num_vertex;
    const int sz_block;
    const int idx_bytes;
    const int data_bytes;

    uint64_t hbm_start_addr;
    uint64_t hbm_end_addr;

    uint64_t input_start_addr;
    uint64_t input_end_addr;

    uint64_t output_start_addr;
    uint64_t output_end_addr;

    uint64_t weight_start_addr;
    uint64_t weight_end_addr;

    uint64_t edge_start_addr;
    uint64_t edge_end_addr;


    const int space_edge_chunk;
    int num_edge_chunk;
    std::vector<int> end_vertex_of_echunk;
    std::vector<int> start_vertex_of_echunk;
    std::vector<std::vector<std::vector<int>>> edge_chunk;

    std::shared_ptr<TileConfig> tile_config;

    void TaskReset(int num_chunk, const std::vector<int>& start,
                   const std::vector<int> end, const std::vector<std::vector<std::vector<int>>>& echunk) {
        num_edge_chunk = num_chunk;
        start_vertex_of_echunk = start;
        end_vertex_of_echunk = end;
        edge_chunk = echunk;
    }



};



#endif //DRAMSIM3_CONFIG_H
