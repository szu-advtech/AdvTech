//
// Created by berry on 2022/7/7.
//


#undef NDEBUG
#include <fstream>
#include <vector>
#include <iostream>
#include <assert.h>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <functional>

#include <tool/timer.h>
#include "configuration.h"
#include "hmc.h"


using namespace dramsim3;
const std::string output_dir = ".";
const std::string config_file = "configs/HMC-Based-8GB.ini";
const std::string fpim_config_path = "configs/GCIM.ini";
const std::string graph_path = "gcn_dataset";

//const std::vector<std::string> gcns = {"gcn"};
//const std::vector<std::string> graphs = {"cora"};
const std::vector<std::string> gcns = {"gcn"};
const std::vector<std::string> graphs = {"cora", "citeseer", "dblp", "pubmed"};
void hmc_callback(uint64_t addr) {
    return;
}

int main() {

    SpMatrix sp;
    sp.LoadSpMatrix(3,4, "gcn_dataset/atest.csv");

    SpMatrix sp2 = std::move(sp);
    return 0;



//    const std::string graph_name = "cora";
//    const std::string gcn_name = "gcn";
//
//    Config config(config_file, output_dir);
//    const std::string layer_path = "gcn/"+ gcn_name + ".ini";
//    config.InitFPIM(fpim_config_path, layer_path);
//    config.fpim_config->LoadGraph(graph_path, graph_name);
////    auto loader = config.fpim_config->graph_loader;
////    Graph graph = loader.raw_graph;
//
//    const int channel_id = 0;
//    const int group_id = 0;
//    dramsim3::Address addr;
//    addr.channel = channel_id;
//    addr.bankgroup = group_id;
//    GroupLogic group_logic(config, addr);
//
//    auto &fpim_config = config.fpim_config;
//    std::vector<BankMemLayout> bank_layouts;
//
//    const auto &csr = fpim_config->graph_loader.raw_graph.r_adj;
//    int start_vertex = fpim_config->mapper.GetGroupStartVertex(channel_id, group_id);
//    int end_vertex = fpim_config->mapper.GetGroupEndVertex(channel_id, group_id);
//    int vertices_per_bank = std::ceil((double)(end_vertex - start_vertex) / config.dram_struct.banks_per_group);
//    int edge_page = 1;
//    std::vector<std::vector<StorCell>> compress_edges;
//
//    if(group_logic.group_type == GroupType::COMPUTE) {
//        auto edges = std::vector<std::vector<int>>(csr.begin() + start_vertex, csr.begin() + end_vertex);
//        compress_edges = fpim_config->edge_compressor.GenEdgeLayout(edges, start_vertex);
//        edge_page = compress_edges.size();
//    }
//
//    for(size_t j = 0; j < config.banks_per_group; j++) {
//        int bank_id = group_id * config.banks_per_group + j;
//        int edge_page_per_bank = edge_page / config.banks_per_group + 1;
//        BankType bank_type = group_logic.group_type == GroupType::COMPUTE ?
//                             BankType::COMPUTE : BankType::STORAGE;
//        bank_layouts.emplace_back(channel_id, bank_id, bank_type);
//        bank_layouts.back().Init(*fpim_config, vertices_per_bank, edge_page_per_bank);
//    }
//
//    int start_bank = group_id * config.banks_per_group;
//    int end_bank = start_bank + config.banks_per_group;
//
//    group_logic.Init(compress_edges,
//                        std::vector<BankMemLayout>(bank_layouts.begin() + start_bank, bank_layouts.begin() + end_bank));
//
//    uint64_t clk = 0;
//    while(true) {
//        group_logic.ClockTick();
//        if (!group_logic.non_cpt_queue.empty() && clk%128 ==1) {
//            auto event = group_logic.non_cpt_queue.front();
//            group_logic.non_cpt_queue.pop_front();
//            group_logic.return_queue.push_back(event);
//
//        }
//        if (!group_logic.cpt_queue.empty() && clk%64 ==0) {
//            group_logic.return_queue.push_back(group_logic.cpt_queue.front());
//            group_logic.cpt_queue.pop_front();
//        }
//        if (group_logic.IsDone()) {
//            break;
//        }
//        clk++;
//    }
//
//    return 0;
}