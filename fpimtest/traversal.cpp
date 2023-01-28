//
// Created by berry on 2022/6/24.
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



enum class VType{
    HUB,
    NON_HUB,
    UNCERTAIN
};
class VSet {
public:
    bool visited;
    VType type = VType::UNCERTAIN;
    int vid;
    int num_vertex;
    uint64_t num_edge; // (undirected graph)
    uint64_t num_iedge; // the number of the intra-set edges
    uint64_t num_est_iedge; // the number of the intra-set edges (estimated)
};

class Detector {
public:
    Detector(const Graph& graph): fset(graph.num_vertex), n_vertex(graph.num_vertex) {
        vsets.resize(graph.num_vertex);
        for(int i = 0; i < n_vertex; i++) {
            vsets[i].visited = false;
            vsets[i].type = VType::UNCERTAIN;
            vsets[i].vid = i;
            vsets[i].num_vertex = 1;
            vsets[i].num_edge = graph.r_adj[i].size();
            vsets[i].num_iedge = 1;
            vsets[i].num_est_iedge = 1;
        }
    }
    int Find(int vid) {
        return fset.Find(vid);
    }
    bool Connect(int va, int vb) {
        return fset.Connect(va, vb);
    }
    bool IsRoot(int vid) {
        return vid == fset.Find(vid);
    }
    VSet& GetVSet(int vid) {
        return vsets[vid];
    }
private:
    UnionFind fset;
    std::vector<VSet> vsets;
    int n_vertex;
};

void Traverse(Graph &graph) {
    int n_vertex = graph.num_vertex;
    int n_edge = graph.num_edge;
    std::cout << "n_vertex:" << n_vertex << ", n_edge:" << n_edge << std::endl;

    std::vector<int> degree;
    degree.resize(n_vertex);
    Detector detector(graph);
    for(int i = 0; i < n_vertex; i++) {
        degree[i] = graph.r_adj[i].size();
    }


    auto sorted_idx = SortIdx(degree, false);


    const int n_group = 64;
    const double iedge_ratio = 0.5;
    const int avg_vertex = graph.num_vertex / n_group;
    const uint64_t avg_edge = graph.num_edge / n_group;
    const int edge_per_vertex =  std::ceil((double)graph.num_edge / graph.num_vertex);
    const int hub_cap = 1024;

    Timer timer;

    const int thresh = 100;

    int cnt = 0;
    for(int i = 0; i < n_vertex; i++) {
        int u = sorted_idx[i];
        bool is_hub = false;
        int est_iedge_u = 0;
        int sum_est_iedge = 0;
        int sum_vertex = 1;
        int sum_edge = detector.GetVSet(u).num_edge;
        std::unordered_set<int> neighbor_set;
        std::unordered_map<int, uint64_t> hub_set;


        {
            for (int v : graph.r_adj[u]) {
                int root_v = detector.Find(v);
                const auto &vset_v = detector.GetVSet(root_v);
                auto iter = neighbor_set.find(root_v);
                switch(vset_v.type) {
                    case VType::NON_HUB: {
                        est_iedge_u++;
                        if (iter == neighbor_set.end()) {
                            neighbor_set.insert(root_v);
                            sum_vertex += vset_v.num_vertex;
                            sum_edge += vset_v.num_edge;
                            sum_est_iedge += vset_v.num_est_iedge;
                        }
                    }
                    break;
                    case VType::HUB: {
                        auto hub_iter = hub_set.find(root_v);
                        if (hub_iter == hub_set.end()) {
                            hub_set.emplace(std::make_pair(root_v,1));
                        } else {
                            hub_iter->second++;
                        }

                    }
                    break;
                }
            }

            int degree_u = degree[u];
            bool case1 = (degree_u > edge_per_vertex || degree_u > thresh ) && est_iedge_u < iedge_ratio * degree_u; // limitation1, hubs ?
            bool case2 = sum_vertex > 1.2 * avg_vertex || sum_edge > 1.2 * avg_edge; // limitation2, nums of edge and vertex ?
            if(case1 || case2) {
                is_hub = true;
            }
        }

        //update

        if (is_hub) {
            auto &vset_u = detector.GetVSet(u);
            vset_u.visited = true;
            vset_u.type = VType::HUB;
            sum_est_iedge = 0;
            sum_vertex = 1;
            sum_edge = detector.GetVSet(u).num_edge;

            for(auto pair_v : hub_set) {
                int v = pair_v.first;
                auto &hub_v = detector.GetVSet(v);
                ASSERT( hub_v.type == VType::HUB);
                if (hub_v.num_vertex + sum_vertex < hub_cap) {
                    sum_est_iedge += pair_v.second;
                    sum_vertex += hub_v.num_vertex;
                    sum_edge += hub_v.num_edge;
                    detector.Connect(u, v);
                }
            }

        } else {
            auto &vset_u = detector.GetVSet(u);
            vset_u.visited = true;
            vset_u.type = VType::NON_HUB;

            for(int v : neighbor_set) {
                detector.Connect(u, v);
                ASSERT(detector.GetVSet(v).type == VType::NON_HUB);
            }
        }
        int u_root = detector.Find(u);
        auto &vset_uroot = detector.GetVSet(u_root);
        vset_uroot.num_vertex = sum_vertex;
        vset_uroot.num_edge = sum_edge;
        vset_uroot.num_est_iedge = est_iedge_u + sum_est_iedge;

    }

    timer.duration("traversal");

    std::unordered_set<int> hubs;
    std::unordered_map<int, std::vector<int>> comm;
    for(int i = 0; i < n_vertex; i++) {
        const auto &set = detector.GetVSet(i);
        if (set.type == VType::HUB) {
            int hub_id = set.vid;
            if(detector.Find(hub_id) == hub_id) {
                hubs.insert(hub_id);
                std::cout << "hub_set_id:" << hub_id << ", hub_size:" << set.num_vertex << ",hub_edge:" << set.num_edge << std::endl;
            }

//            std::cout << "hub: " << hub_id << ", degree: " << degree[hub_id] << std::endl;
        } else if (set.type == VType::NON_HUB) {
            int root_id = detector.Find(i);
            auto iter = comm.find(root_id);
            if ( comm.end() == iter) {
                comm.emplace(root_id, std::vector<int>{i});

            } else {
                iter->second.push_back(i);
            }
            if (detector.IsRoot(i)) {
//                std::cout << "vertex:" << vset[i].num_vertex << ", edge: " << vset[i].num_edge << ", iedge: " << vset[i].num_est_iedge << std::endl;
            }
        }
    }
    std::cout << "hub: " << hubs.size() << std::endl;
    std::cout << "comm: " << comm.size() << std::endl;

//    std::unordered_set<int> high_degree_set;
//    for(int i = 0; i < n_vertex; i++) {
//        if (degree[i] > edge_per_vertex) {
//            high_degree_set.insert(i);
//        }
//    }
//    std::cout << "high: " << high_degree_set.size() << std::endl;

//    for (auto pair : comm) {
//        int root_id = pair.first;
//        auto &members = pair.second;
//        std::unordered_set<int> local_set{members.begin(), members.end()};
//        std::unordered_set<int> hub_set;
//        int hub_edge = 0;
//        int num_iedge = 0;
//        for (auto u : members) {
//            for (auto v : graph.r_adj[u]) {
//                if (local_set.find(v) != local_set.end()) {
//                    num_iedge++;
//                } else {
//                    if(hubs.find(v) != hubs.end()) {
//                        hub_set.emplace(v);
//                        hub_edge++;
//                    }
//                }
//            }
//        }
//        auto &vset_root = detector.GetVSet(root_id);
//        vset_root.num_iedge = num_iedge;
//        if (vset_root.num_vertex > 0) {
//            std::cout << "vertex:" << vset_root.num_vertex << ", edge: " << vset_root.num_edge << ", est_iedge: " << vset_root.num_est_iedge
//                      << ", iedge: " << vset_root.num_iedge << ", hub_vertex?:" << hub_set.size() << ", hub_edge:" << hub_edge << std::endl;
//        }
//
//    }

}
bool cmp(const int& lhs, const int& rhs){
    return lhs > rhs;
}

int main() {

    const std::string graph_name = "reddit";
    const std::string gcn_name = "gcn";

    Config config(config_file, output_dir);
    const std::string layer_path = "gcn/"+ gcn_name + ".ini";
    config.InitFPIM(fpim_config_path, layer_path);
    config.fpim_config->LoadGraph(graph_path, graph_name);
    auto loader = config.fpim_config->graph_loader;
    Graph graph = loader.raw_graph;

//    Traverse(graph);
    uint64_t n_vertex = graph.num_vertex;
    uint64_t n_edge = graph.num_edge;
    std::vector<int> degree;
    degree.resize(n_vertex);
    Detector detector(graph);

    uint64_t edges = 0;
    uint64_t vertices = 0;
    for(int i = 0; i < n_vertex; i++) {
        degree[i] = graph.r_adj[i].size();
        if(degree[i] > 5) {
            vertices++;
            edges += degree[i];
        }
    }
    std::sort(degree.begin(), degree.end(),cmp);
    return 0;


}