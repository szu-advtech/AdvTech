//
// Created by szu on 2021/11/16.
//

#include "graph.h"
#include <cassert>
#include <fstream>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include "tool.h"

void SpMatrix::LoadSpMatrix(int h, int w, const std::string file_path) {
    height = h;
    width = w;
    adj.resize(h);
    r_adj.resize(w);

    std::ifstream sparse(file_path);
    ASSERT(sparse.is_open());

    std::string line;
    const std::string delimiter = ",";
    while (std::getline(sparse, line)) {;
        auto pos = line.find(delimiter);
        int src = std::atoi(line.substr(0, pos).c_str());
        int dst = std::atoi(line.substr(pos + 1, line.size()).c_str());


        if (!(dst < w && src < h)) {
            ASSERT(src < h);
            ASSERT(dst < w);
        }
        r_adj[dst].push_back(src);
        adj[src].push_back(dst);

    }
    sparse.close();
}

void SpMatrix::GenFullMatrix(int h, int w) {
    height = h;
    width = w;
    adj.resize(h);
    r_adj.resize(w);
    for (int i = 0; i < h; i++) {
        adj[i].resize(w);
        for (int j = 0; j < w; j++) {
            adj[i][j] = j;
        }
    }

    for (int i = 0; i < w; i++) {
        r_adj[i].resize(h);
        for (int j = 0; j < h; j++) {
            r_adj[i][j] = j;
        }
    }

}

void GraphLoader::LoadGraph(const std::string &path, const std::string &graph_name) {
    raw_graph = std::make_shared<Graph>();
    this->graph_name = graph_name;
    raw_graph->graph_name = graph_name;

    if (graph_name == "citeseer"    ||
        graph_name == "cora"        ||
        graph_name == "pubmed"      ||
        graph_name == "reddit"      ||
        graph_name == "dblp"        ||
        graph_name == "test"        ) {
    } else {
        ASSERT(false);
    }
    std::string graph_info_path = path + "/" + graph_name + ".txt";
    std::string graph_edge_path = path + "/" + graph_name + "_edge.csv";
    std::ifstream graph_info(graph_info_path);
    ASSERT(graph_info.is_open());

    int &num_edge = raw_graph->num_edge;
    int &num_vertex = raw_graph->num_vertex;
    int &num_class = raw_graph->num_class;
    int &len_feature = raw_graph->len_feature;

    std::string line;
    const std::string delimiter = ",";
    while(std::getline(graph_info,line)) {
        auto pos = line.find(delimiter);
        const std::string key = line.substr(0, pos).c_str();
        int val = std::atoi(line.substr(pos + 1, line.size()).c_str());

        if(key == "edge") {
            num_edge = val;
        } else if(key == "vertex") {
            num_vertex = val;
        } else if(key == "feature") {
            len_feature = val;
        } else if(key == "class" ) {
            num_class = val;
        } else {
            ASSERT(false);
        }
    }
    graph_info.close();
    ASSERT(num_edge != -1);
    ASSERT(num_vertex != -1);
    ASSERT(num_class != -1);
    ASSERT(len_feature != -1);

    raw_graph->LoadSpMatrix(num_vertex, num_vertex, graph_edge_path);

}

Graph GraphLoader::GetGcnNormGraph() {
    Graph new_graph;
    int edge = 0;
    new_graph.num_vertex = raw_graph->num_vertex;
    new_graph.num_class= raw_graph->num_class;
    new_graph.len_feature = raw_graph->len_feature;
    new_graph.graph_name = raw_graph->graph_name;
    new_graph.height = raw_graph->height;
    new_graph.width = raw_graph->width;

    new_graph.r_adj.resize(new_graph.num_vertex);

    for(int i = 0; i < new_graph.num_vertex; i++) {
        auto &src = raw_graph->r_adj[i];
        std::set<int> set(std::make_move_iterator(src.begin()),
                          std::make_move_iterator(src.end()));
        set.insert(i);
        new_graph.r_adj[i] = std::vector<int>(std::make_move_iterator(set.begin()),
                                              std::make_move_iterator(set.end()));
        edge += new_graph.r_adj[i].size();
    }

    new_graph.num_edge = edge;

    return new_graph;
}

Graph GraphLoader::GetSampleGraph(int sample_size) {

    std::random_device rd; // Non-deterministic seed source
    std::default_random_engine rng {rd()}; // Create random number generator

    Graph new_graph;
    new_graph.num_edge = sample_size * raw_graph->num_vertex;
    new_graph.num_vertex = raw_graph->num_vertex;
    new_graph.num_class= raw_graph->num_class;
    new_graph.len_feature = raw_graph->len_feature;
    new_graph.graph_name = raw_graph->graph_name;
    new_graph.height = raw_graph->height;
    new_graph.width = raw_graph->width;

    new_graph.r_adj.resize(new_graph.num_vertex);
    for(int i = 0; i < new_graph.num_vertex; i++) {
        int dst = i;
        const auto &srcs = raw_graph->r_adj[dst];
        int num_src = srcs.size();
        if (num_src == 0) {
            continue;
        }
        std::uniform_int_distribution<int> range {0, num_src - 1};
        for(int j = 0; j < sample_size; j++) {
            int src_idx = range(rng);
            int src = srcs[src_idx];
            new_graph.r_adj[dst].push_back(src);
        }
        std::sort(new_graph.r_adj[dst].begin(), new_graph.r_adj[dst].end());
    }

    // test
    ASSERT(new_graph.r_adj.size() == raw_graph->r_adj.size());
    ASSERT(new_graph.adj.size() == 0);
    for(int i = 0; i < new_graph.num_vertex; i++) {
        ASSERT (new_graph.r_adj[i].size() == sample_size ||
                new_graph.r_adj[i].size() == 0);
    }

    const std::string out_path = "sample/" + graph_name + "_sample.csv";
    new_graph.DumpEdge(out_path);

    return new_graph;
}

Graph GraphLoader::LoadSampleGraph(int sample_size) {
    Graph new_graph;
    new_graph.num_edge = sample_size * raw_graph->num_vertex;
    new_graph.num_vertex = raw_graph->num_vertex;
    new_graph.num_class= raw_graph->num_class;
    new_graph.len_feature = raw_graph->len_feature;
    new_graph.graph_name = raw_graph->graph_name;
    new_graph.height = raw_graph->height;
    new_graph.width = raw_graph->width;


    new_graph.r_adj.resize(new_graph.num_vertex);

    std::string graph_edge_path = "sample/" + graph_name + "_sample.csv";
    std::ifstream graph_edge(graph_edge_path);

    std::string line;
    const std::string delimiter = ",";
    while (std::getline(graph_edge, line)) {
        auto pos = line.find(delimiter);
        int src = std::atoi(line.substr(0, pos).c_str());
        int dst = std::atoi(line.substr(pos + 1, line.size()).c_str());
        new_graph.r_adj[dst].push_back(src);
    }
    return new_graph;

}

void GraphLoader::Shuffle(std::vector<std::vector<int>>& r_adj) {
    for(int i = 0; i < r_adj.size(); i++) {
        auto &srcs = r_adj[i];
        // unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
        std::shuffle (srcs.begin (), srcs.end (), std::default_random_engine ());
    }
}

HashGraph::HashGraph(const Graph& graph) {
    ASSERT(graph.adj.size() != 0 && graph.num_vertex != 0);
    num_vertex = graph.num_vertex;
    num_edge = graph.num_edge;
    num_class = graph.num_class;
    len_feature = graph.len_feature;

    adj.resize(num_vertex);

    for(int src = 0; src < graph.num_vertex; src++) {
        for(auto dst: graph.adj[src]) {
            adj[src].insert(dst);
        }
    }

    // test
    for(int src = 0; src < graph.num_vertex; src++) {
        ASSERT(adj[src].size() == graph.adj[src].size());
        for(auto dst: graph.adj[src]) {
            ASSERT(adj[src].find(dst) != adj[src].end());
        }
    }

    is_directed = false;
    for(int src = 0; src < graph.num_vertex; src++) {
        for(auto dst: graph.adj[src]) {
            if(adj[dst].find(src) == adj[dst].end()) {
                is_directed = true;
                return;
            }
        }
    }
}

void HashGraph::ToUndirected() {
    for(int src = 0; src < num_vertex; src++) {
        for(auto dst : adj[src]) {
            adj[dst].insert(src);
        }
    }
    int edge = 0;
    for(int src = 0; src < num_vertex; src++) {
        edge += adj[src].size();
    }
    num_edge = edge;
    is_directed = false;
}

void GraphLoader::Reorder(Graph &graph, std::unordered_map<int, int>& reorder_map) {
    int num_vertices =  graph.num_vertex;
    std::vector<std::vector<int>> a;
    std::vector<std::vector<int>> r_a;
    a.resize(num_vertices);
    r_a.resize(num_vertices);
    for(int v = 0; v < num_vertices; v++) {
        int new_v = reorder_map[v];
        a[new_v].reserve(graph.r_adj[v].size());
        for(auto e : graph.r_adj[v]) {
            a[new_v].push_back(reorder_map[e]);
        }

    }
    for(int v = 0; v < num_vertices; v++) {
        int new_v = reorder_map[v];
        r_a[new_v].reserve(graph.r_adj[v].size());
        for(auto e : graph.r_adj[v]) {
            r_a[new_v].push_back(reorder_map[e]);
        }
    }
    graph.adj = a;
    graph.r_adj = r_a;
}



std::vector<std::vector<StorCell>> EdgeCompress::GenEdgeLayout(std::vector<std::vector<int>> csr, int vertex_start_idx) {
    edge_pages_.clear();
    for (int i = 0; i < csr.size(); i++) {
        int row = vertex_start_idx + i;
        FillEdgePage(row,  row, true);
        for (int j = 0; j < csr[i].size(); j++) {
            FillEdgePage(row, csr[i][j]);
        }
    }
    // Sentinel
    FillEdgePage(-1, -1, true, true);
    return edge_pages_;
}


void EdgeCompress::FillEdgePage(int row, int col, bool vertex_head, bool over) {

    // full page
    int ele_per_page = sz_page / (data_bytes + idx_bytes);
    if (current_page_cells_.size() == ele_per_page) {
        edge_pages_.push_back(current_page_cells_);
        current_page_cells_.clear();
    }

    // new page or new row
    if (current_page_cells_.empty() || vertex_head) {
        current_page_cells_.emplace_back(row);
        current_page_cells_.emplace_back(-1); // value
    }

    // general case
    if (!vertex_head) {
        current_page_cells_.emplace_back(col);
        current_page_cells_.emplace_back((float)0.0); // the value
    }

    if (over) {
        edge_pages_.push_back(current_page_cells_);
        current_page_cells_.clear();
        return;
    }

    ASSERT(current_page_cells_.size() <= sz_page);
}


