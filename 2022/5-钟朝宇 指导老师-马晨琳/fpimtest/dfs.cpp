//
// Created by berry on 2022/9/1.
//
#undef NDEBUG
#include <vector>
#include <unordered_map>
#include <functional>
#include <cassert>

#include "tool/timer.h"
#include "configuration.h"
#include "hmc.h"
#include "tool/tool.h"

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

class Traversal {
public:
    const Graph& graph;
    std::vector<bool> visited;
    std::vector<int> allocation;

    int cap_edge = 0;
    const int total_edge;
    const int total_vertex;
    const int num_group;
    const int cap_idx;
    const int degree_thresh;
    int base_load;
    int group_load;


    std::vector<std::vector<int>> base_batch;
    std::vector<std::unordered_set<int>> base_comm;
    std::vector<int> base_edge;

    std::vector<std::vector<int>> group_batch;
    std::vector<std::unordered_set<int>> group_comm;
    std::vector<int> group_edge;

    Traversal(const Graph &graph, int cim_group, int degree_thresh):
        graph(graph), total_edge(graph.num_edge),
        total_vertex(graph.num_vertex), num_group(cim_group),
        cap_idx(UpperAlign(total_vertex, num_group)),
        degree_thresh(degree_thresh)
    {
        base_load = 0;
        auto degrees = graph.GetDegrees();
        auto sorted_idx = graph.GetSortedVertex(degrees);
        int vertex_limit = 512;

        int base_iter_cnt = 0;

        base_batch.emplace_back();
        base_comm.emplace_back();
        base_edge.emplace_back(0);
        for (auto idx : sorted_idx) {
            int degree = degrees[idx];
            if (degree < degree_thresh) {
                break;
            }
            base_load += degree;

            base_batch[base_iter_cnt].push_back(idx);
            base_edge[base_iter_cnt] += graph.r_adj[idx].size();
            for (auto neighbor : graph.r_adj[idx]) {
                base_comm[base_iter_cnt].insert(neighbor);
            }

            if (base_batch[base_iter_cnt].size() >= vertex_limit) {
                base_iter_cnt++;
                base_batch.emplace_back();
                base_comm.emplace_back();
                base_edge.emplace_back(0);
            }
        }

        for (int i = 0; i < base_batch.size(); i++) {
            std::cout << "base_iter:" << i << ", comms:" << base_comm[i].size() << std::endl;
        }

        group_load = graph.num_edge - base_load;
        cap_edge = std::ceil((double)group_load / num_group);

    }

    void Traverse() {
        int iter_cnt = 0;
        visited.clear();
        visited.resize(total_vertex, false);
        allocation.clear();
        allocation.resize(total_vertex, -1);

        group_batch.clear();
        group_batch.resize(num_group);
        group_comm.clear();
        group_comm.resize(num_group);
        group_edge.clear();
        group_edge.resize(num_group, 0);


        for (int i = 0; i < cap_idx; i++) {
            int vid = NextVertex(i);
            if (vid < total_vertex && !visited[vid]) {
                BDFS(vid, 6, iter_cnt);
                if (group_edge[iter_cnt] > cap_edge) {
                    iter_cnt++;
                }
            }
        }

        for(int i = 0; i < num_group; i++) {
            std::cout << "iter" << i << ", batch:" << group_batch[i].size()
                      << ", comm:" << group_comm[i].size() << ", edge:" << group_edge[i] << std::endl;
        }

        int addup = 0;
        for(auto &batch : base_batch) {
            addup += batch.size();
        }
        for(auto &batch : group_batch) {
            addup += batch.size();
        }
        if(addup != graph.num_vertex) {
            std::cout << "addup: " << addup << ", while: " << graph.num_vertex << std::endl;
            ASSERT(addup == graph.num_vertex);
        }
    }

    void Analyze() {

        int len_feature = 128;
        const int data_byte = 4;
        const int base_bandwidth = 128;
        const int group_bandwidth = 128 / num_group;
        const uint64_t base_cpt_ability = 128; // 128 op/ns;
        const uint64_t group_cpt_ability = 256 / num_group; // 8 op/ns;


        uint64_t base_fetch_vertices = 0;
        uint64_t base_cpt_edges = 0;
        for (int i = 0; i < base_batch.size(); i++) {
            base_fetch_vertices += base_comm[i].size();
            base_cpt_edges += base_edge[i];
        }
        uint64_t mem_base_latency = (double)base_fetch_vertices * len_feature * data_byte / base_bandwidth;
        uint64_t cpt_base_latency = base_cpt_edges * len_feature / base_cpt_ability;
        std::cout << "base" << ", mem:" << mem_base_latency << ", cpt:" << cpt_base_latency << ", total:" << mem_base_latency + cpt_base_latency << std::endl;

        for (int i = 0; i < group_batch.size(); i++) {
            uint64_t group_fetch_vertices = group_comm[i].size();
            uint64_t group_cpt_edges = group_edge[i];
            uint64_t mem_group_latency = (double) group_fetch_vertices * len_feature * data_byte / group_bandwidth;
            uint64_t cpt_group_latency = group_cpt_edges * len_feature / group_cpt_ability;
            std::cout << "group:" << i << ", mem:" << mem_group_latency << ", cpt:" << cpt_group_latency << ", total:" << cpt_group_latency + mem_group_latency << std::endl;
        }

        return;

    }

    void Optimize() {
        std::vector<double> avg_ratio; // comms per edge
        for (int i = 0; i < group_batch.size(); i++) {
            int comm = group_comm[i].size();
            uint64_t n_edge = group_edge[i];
            int n_vertex = group_batch[i].size();
            std::cout << "cpe:" << (double)comm / n_edge << ", cpv:" << (double)comm / n_vertex << std::endl;
        }

        int b_comm = 0;
        uint64_t b_edge = 0;
        int b_vertex = 0;
        for (int i = 0; i < base_batch.size(); i++) {
            int tmp_comm = base_comm[i].size();
            uint64_t tmp_edge = base_edge[i];
            int tmp_vertex = base_batch[i].size();
            std::cout << "base: cpe:" << (double)tmp_comm / tmp_edge << ", cpv:" << (double)tmp_comm / tmp_vertex << std::endl;

            b_comm += tmp_comm;
            b_edge += tmp_edge;
            b_vertex += tmp_vertex;
        }
        std::cout << "cpe:" << (double)b_comm / b_edge << ", cpv:" << (double)b_comm / b_vertex << std::endl;

    }

    void Test() {
        std::vector<std::vector<std::vector<int>>> sp_input;
        std::vector<std::unordered_map<int,int>> sp_hash;
        std::vector<std::vector<std::vector<int>>> base_input;
        std::vector<std::unordered_map<int,int>> base_hash;

        sp_input.resize(num_group);
        sp_hash.resize(num_group);

        base_input.resize(base_batch.size());
        base_hash.resize(base_batch.size());

        for (int i = 0; i < num_group; i++) {
            auto &batch = group_batch[i];
            auto &sp = sp_input[i];
            auto &hash = sp_hash[i];

            sp.resize(total_vertex);
            for (int j = 0; j < batch.size(); j++) {
                int row = batch[j];
                hash[j] = row;
                for(auto col : graph.r_adj[row]) {
                    sp[col].push_back(j);
                }
            }
        }

        for (int i = 0; i < base_batch.size(); i++) {
            auto &batch = base_batch[i];
            auto &sp = base_input[i];
            auto &hash = base_hash[i];

            sp.resize(total_vertex);
            for (int j = 0; j < batch.size(); j++) {
                int row = batch[j];
                hash[j] = row;
                for(auto col : graph.r_adj[row]) {
                    sp[col].push_back(j);
                }
            }
        }
    }

private:
    int NextVertex(int idx) {
        const static int sz_chunk = cap_idx / num_group;
        int chunk_id = idx % num_group;
        int logic_id = idx / num_group;

        return chunk_id * sz_chunk + logic_id;
    }
    void BDFS(int src_id, int depth, int iter) {
        if (group_edge[iter] > cap_edge) {
            return;
        }
        visited[src_id] = true;
        allocation[src_id] = iter;
        int degree = graph.r_adj[src_id].size();
        if (degree < degree_thresh) {
            group_batch[iter].push_back(src_id);
            group_edge[iter] += graph.r_adj[src_id].size();
        }
        for(int dst_id : graph.r_adj[src_id]) {
            if (degree < degree_thresh) {
                group_comm[iter].insert(dst_id);
            }
            if (!visited[dst_id] && depth > 0) {
                BDFS(dst_id, depth - 1, iter);
            }
        }
    }
};

//void kkp() {
//    int pim_edge;
//    int non_pim_edge;
//
//    int data_byte = 4;
//    int idx_byte = 4;
//    int len_feature = 128;
//    int num_group = 32;
//    int edge = 10;
//    double power_dy_abuf;
//    double power_dy_bbuf;
//    double power_dy_mac = 1.9140; //mw
//    double energy_per_mac_op = power_dy_mac * 2 / 1000; //nj
//    double energy_per_iread = 0.3447 / 32; // nj 32bytes
//    double energy_per_iwrite = 0.3449 / 32; // nj 32bytes
//    double energy_per_dram_read = 2.01618 / 32; // nj 32bytes
//    double energy_per_dram_write = 2.01633 / 32; // nj 32bytes
//    double energy_per_sram_read = 0.00165861 / 16; // nj 16bytes
//    double energy_per_sram_write = 0.00213977 / 16; // nj 16bytes;
//    double energy_dy_cim_op = energy_per_mac_op + (energy_per_iread + energy_per_iwrite) * data_byte; // nj per_op
//
//    int byte_sram_read_abuf = edge * (idx_byte * 2 + data_byte);
//    int byte_sram_read_bbuf = edge * len_feature * data_byte;
//    int byte_sram_write_abuf = byte_sram_read_abuf;
//    int byte_sram_write_bbuf = len_feature * data_byte * //avereage
//    int byte_dram_read = byte_sram_write_abuf;
//    return;
//}



int main() {
    const std::string graph_name = "cora";
    const std::string gcn_name = "gcn";

    Config config(config_file, output_dir);
    const std::string layer_path = "gcn/"+ gcn_name + ".ini";
    config.InitFPIM(layer_path);
    config.fpim_config->LoadGraph(graph_path, graph_name, false);
    auto loader = config.fpim_config->graph_loader;
    uint64_t base_load = loader.raw_graph->num_edge * 0.5;
    int degree_thresh = loader.raw_graph->GetDegreeThresh(base_load);


    Timer timer;
    Traversal traversal(*(loader.raw_graph), 32, degree_thresh);
    traversal.Traverse();
    timer.duration("traversal");

    //traversal.Analyze();
    //traversal.Optimize();
    traversal.Test();


    return 0;


}

