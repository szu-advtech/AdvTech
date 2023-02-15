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
#include "logic/base_logic.h"


using namespace dramsim3;
const std::string output_dir = ".";
const std::string config_file = "configs/HMC-Based-8GB.ini";
const std::string graph_path = "gcn_dataset";

const std::vector<std::string> gcns = {"gcn","gin","gs"};
//const std::vector<std::string> graphs = {"cora", "dblp", "reddit", "pubmed"};
//const std::vector<std::string> gcns = {"gs"};
const std::vector<std::string> graphs = {"cora"};//"dblp", "cora"};//,, "citeseer", "reddit"
void hmc_callback(uint64_t addr) {
    return;
}


void DummyTest() {
//    const std::string graph_name = "citeseer";
//    const std::string gcn_name = "gcn";
    for(const auto gcn_name : gcns) {
        for(const auto graph_name : graphs) {
            Config config(config_file, output_dir);
            const std::string layer_path = "gcn/"+ gcn_name + ".ini";
            config.InitFPIM( layer_path);
            config.fpim_config->LoadGraph(graph_path, graph_name);

            uint64_t clk = 0;
            Timer timer;

            int channel_id = 0;
            int group_id = 0;
            std::vector<std::vector<int>> dummy_sp;
            std::unordered_map<int, int> dummy_map;


            int cim_groups = 2; //config.channels * config.dram_struct.cim_group_per_channel;
            int num_vertex = config.fpim_config->graph_loader.raw_graph->num_vertex;
            int vertex_per_group = std::ceil((double) num_vertex / cim_groups);

            const auto& sp = config.fpim_config->layer_config.GetCurrentKernel().sp->r_adj;
            int sp_width = config.fpim_config->layer_config.GetCurrentKernel().dim_b;
            int sp_height = config.fpim_config->layer_config.GetCurrentKernel().dim_a;

            std::vector<std::vector<std::vector<int>>> sp_partition;
            std::vector<std::unordered_map<int,int>> sp_hash;
            sp_partition.resize(cim_groups);
            sp_hash.resize(cim_groups);
            for (int i = 0; i < cim_groups; i++) {
                sp_partition[i].resize(sp_width);
            }
            for (int i = 0; i < sp_height; i++) {
                int group_id = i / vertex_per_group;
                int logic_vid = i % vertex_per_group;
                sp_hash[group_id][logic_vid] = i;
            }
            for (int i = 0; i < sp_width; i++) {
                for(auto vid : sp[i]) {
                    int group_id = vid / vertex_per_group;
                    int logic_vid = vid % vertex_per_group;
                    sp_partition[group_id][i].push_back(logic_vid);
                }
            }

            for (int i = 0; i < cim_groups; i++) {
                clk = 0;
                GroupLogic group_logic(config, channel_id, i);
                group_logic.Init(sp_partition[i], sp_hash[i]);

                while(!group_logic.IsDone()) {
                    group_logic.ClockTick();
                    clk++;
                    if(clk % 64 == 0) {
                        while (!group_logic.intra_group_queue.empty()) {
                            auto event = group_logic.intra_group_queue.front();
                            group_logic.intra_group_queue.pop_front();
                            group_logic.return_queue.push_back(event);
                        }
                    }
                    if(clk % 64 == 0) {
                        while (!group_logic.inter_group_queue.empty()) {
                            auto event = group_logic.inter_group_queue.front();
                            group_logic.inter_group_queue.pop_front();
                            group_logic.return_queue.push_back(event);
                        }
                    }

                }
                std::cout << clk << std::endl;
            }

//            group_logic.Init(sp, dummy_map);
//
//            while(!group_logic.IsDone()) {
//                group_logic.ClockTick();
//                clk++;
//                if(clk % 64 == 0) {
//                    while (!group_logic.non_cpt_queue.empty()) {
//                        auto event = group_logic.non_cpt_queue.front();
//                        group_logic.non_cpt_queue.pop_front();
//                        group_logic.return_queue.push_back(event);
//                    }
//                }
//                if(clk % 64 == 0) {
//                    while (!group_logic.cpt_queue.empty()) {
//                        auto event = group_logic.cpt_queue.front();
//                        group_logic.cpt_queue.pop_front();
//                        group_logic.return_queue.push_back(event);
//                    }
//                }
//
//            }




//                break;
            do {
                do {
                } while(config.fpim_config->PrepareNextKernel());
            } while(config.fpim_config->PrepareNextLayer());


//            HMCMemorySystem hmc(config, output_dir, hmc_callback, hmc_callback);
//            hmc.InitFPIM();
//            hmc.Enable(true, true);
//                while(true) {
//                    hmc.ClockTick();
//                    clk++;
//                    if (clk % 1000000 == 0) {
//                        hmc.PrintAggrProgress();
//                    }
//                    if (hmc.IsDone()) {
//                        hmc.PrintStats();
//                        hmc.UpdateHMCStatus();
//                        std::string out_dir = "res/" + gcn_name + "_" + graph_name;
//                        hmc.hmc_status.Dump(out_dir, layer_cnt);
//                        layer_cnt++;
//                        break;
//                    }
//                };
            timer.duration("duration");
            std::cout << "latency: " << clk << std::endl;
            std::cout << "---------" << std::endl;
        }
    }
}

void DummyTest1() {
    for(const auto gcn_name : gcns) {
        for(const auto graph_name : graphs) {
            Config config(config_file, output_dir);
            const std::string layer_path = "gcn/"+ gcn_name + ".ini";
            config.InitFPIM( layer_path);
            config.fpim_config->LoadGraph(graph_path, graph_name);

            uint64_t clk = 0;
            Timer timer;

            BaseLogic base_logic(config);

            base_logic.Init(config.fpim_config->mapper.GetBaseInput(), config.fpim_config->mapper.GetBaseHash());

            while(!base_logic.IsDone()) {
                base_logic.ClockTick();
                clk++;
                if(clk % 64 == 0) {
                    while (!base_logic.extern_queue.empty()) {
                        auto event = base_logic.extern_queue.front();
                        base_logic.extern_queue.pop_front();
                        base_logic.return_queue.push_back(event);
                    }
                }
            }

            timer.duration("duration");
            std::cout << "latency: " << clk << std::endl;
            std::cout << "---------" << std::endl;
        }
    }
}

void Run(const std::string gcn_name ,const std::string graph_name) {
    Config config(config_file, output_dir);
    const std::string layer_path = "gcn/"+ gcn_name + ".ini";
    config.InitFPIM( layer_path);
    config.fpim_config->LoadGraph(graph_path, graph_name);

    uint64_t clk = 0;
    Timer timer;


    int layer_id = 0;
    do {
        do {
            HMCMemorySystem hmc(config, output_dir, hmc_callback, hmc_callback);
            // pim
            hmc.GCNFunctionEnable(true, true);
            // non-pim
            // hmc.GCNFunctionEnable(false, true);
            hmc.InitFPIM();

            while(!hmc.IsDone()) {
                hmc.ClockTick();
                clk++;
                if (clk % 1000000 == 0) {
                    hmc.PrintAggrProgress();
                }
            };
            std::cout << "global clk " << clk << std::endl;
            std::cout << "---- ----" << std::endl;
            hmc.UpdateHMCStatus();
            std::string out_dir = "res/" + gcn_name + "_" + graph_name;
            hmc.hmc_status.Dump(out_dir, layer_id++);

        } while(config.fpim_config->PrepareNextKernel());
    } while(config.fpim_config->PrepareNextLayer());


    timer.duration("duration");
    std::cout << "latency: " << clk << std::endl;
    std::cout << "---------" << std::endl;
}


int main() {
    for(auto gcn_name : gcns) {
        for(auto graph_name : graphs) {
            Run(gcn_name, graph_name);
        }
    }


}