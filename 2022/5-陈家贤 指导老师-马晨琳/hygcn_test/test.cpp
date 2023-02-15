//
// Created by szu on 2021/11/15.
//


#include <iostream>
#include <thread>
#include <memory>
#include <future>
#include <fstream>

#include "hygcn.h"
#include "configuration.h"
#include "config.h"


const std::string sys_path = "configs/HYGCN.ini";
const std::string graph_path = "gcn_dataset";
const std::vector<std::string> gcns = {"gcn", "gin", "gs"};
const std::vector<std::string> graphs = {"cora", "citeseer", "dblp", "pubmed", "reddit"};

void Dump(const std::vector<HyRec>& records, const std::string& dir_path,
          const std::string &gcn_type, const std::string &dataset_name) {
    const std::string name = gcn_type + "_" + dataset_name;
    const std::string output_path = dir_path + "/" + name + ".csv";
    std::ofstream ostrm(output_path);

    ostrm << "name,layer," << HyRec::GetHeaderString();

    for(int i = 0; i < records.size(); i++) {
        ostrm << name + "," + std::to_string(i) + "," + records[i].GetString();
    }
    return;
}


int main() {
    auto hy_config = std::make_shared<HyConfig>();

    for (auto gcn_name : gcns) {
        for (auto graph_name : graphs) {
            const std::string layer_path = "gcn/" + gcn_name + ".ini";
            hy_config->Init(graph_path, graph_name, sys_path, layer_path);
            std::vector<HyRec> records;
            Timer timer1;
            uint64_t clk = 0;
            do {
                HyGCN hygcn(hy_config);
                while(!hygcn.IsDone()) {
                    hygcn.ClockTick();
                    clk++;
                }
                hygcn.Record();
                records.push_back(hygcn.hyrec);
                std::cout << "clk: " << hygcn.clk << std::endl;

            } while (hy_config->PrepareNextLayer());
            Dump(records, "res", gcn_name, graph_name);
            std::cout << "latency: " <<  clk << std::endl;
            timer1.duration("hygcn");
        }
    }
    return 0;
}
