//
// Created by szu on 2021/11/16.
//

#ifndef DRAMSIM3_GRAPH_H
#define DRAMSIM3_GRAPH_H
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <fstream>

#include "tool.h"

// Sparse Matrix
class SpMatrix {
public:
    explicit SpMatrix() {}
    std::vector<std::vector<int>> adj; // record outgoing-edge
    std::vector<std::vector<int>> r_adj; // record incoming-edge
    int height = -1, width = -1;
    virtual void LoadSpMatrix(int h, int w, const std::string file_path);
    virtual void GenFullMatrix(int h, int w);
};

class Graph: public SpMatrix {
public:
    explicit Graph() {}
    int num_vertex = -1;
    int num_edge = -1;
    int num_class = -1;
    int len_feature = -1;
    std::string graph_name;
    void DumpEdge(const std::string out_path) {
        std::ofstream ostrm(out_path);
        for (int i = 0; i < num_vertex; i++) {
            const auto& vertex = r_adj[i];
            for (auto j : vertex) {
                ostrm << std::to_string(i) << "," << std::to_string(j) << std::endl;
            }
        }
    }
    static void SortedDegree(std::vector<int> &degrees) {
        std::sort(degrees.begin(), degrees.end(), [](int a, int b) {return a>b;});
    }

    std::vector<int> GetDegrees() const {
        std::vector<int> degree;
        degree.resize(num_vertex);
        for(int i = 0; i < num_vertex; i++) {
            degree[i] = r_adj[i].size();
        }
        return degree;
    }

    static std::vector<size_t> GetSortedVertex(const std::vector<int>& degrees) {
        return SortIdx(degrees, true);
    }


    int GetDegreeThresh(uint64_t load_edge) const{
        auto degree = GetDegrees();
        SortedDegree(degree);

        int degree_thresh = 0;
        uint64_t load = 0;
        for(int i = 0; i < num_vertex; i++) {
            if (load > load_edge) {
                degree_thresh = degree[i];
                break;
            }
            load += degree[i];
        }
        degree_thresh++;

        return degree_thresh;
    }
};

class HashGraph {
public:
    HashGraph(const Graph& graph);
    std::vector<std::unordered_set<int>> adj; // record outgoing-edge
    void ToUndirected();
    void SelfLoopLess() {
        for(int i = 0; i < num_vertex; i++) {
            adj[i].erase(i);
        }
    }
    int num_vertex = -1;
    int num_edge = -1;
    int num_class = -1;
    int len_feature = -1;
    bool is_directed;
};

class GraphLoader {
public:
    GraphLoader() = default;
    void LoadGraph(const std::string &path, const std::string &graph_name);
    Graph GetGcnNormGraph();
    Graph GetSampleGraph(int sample_size);
    Graph LoadSampleGraph(int sample_size);
    HashGraph GetHashGraph() {
        return HashGraph(*raw_graph);
    }
    static void Shuffle(std::vector<std::vector<int>>& r_adj);
    static void Reorder(Graph &graph, std::unordered_map<int, int>& reorder_map);


    void EliminateEvilRow(int thresh) {
        for (int i = 0; i < raw_graph->num_vertex; i++) {
            if(raw_graph->r_adj[i].size() > thresh) {
                raw_graph->r_adj[i].clear();
            }
        }
    }

    std::shared_ptr<Graph> raw_graph;
private:
    std::string graph_name;
};

class StorCell {
public:
    StorCell(int i) {
        c.i = i;
    }
    StorCell(float f) {
        c.f = f;
    }

    int i() const{
        return c.i;
    }

    float f() const{
        return c.f;
    }

    union Cell{
        int i;
        float f;
    };
    Cell c;
};

class EdgeCompress {
public:
    EdgeCompress(int sz_page, int data_bytes, int idx_bytes):
            sz_page(sz_page), data_bytes(data_bytes), idx_bytes(idx_bytes) {
    }
    std::vector<std::vector<StorCell>> GenEdgeLayout(std::vector<std::vector<int>> csr, int vertex_start_idx);
private:
    void FillEdgePage(int row, int col, bool vertex_head = false, bool over = false);
    std::vector<StorCell> current_page_cells_;
    std::vector<std::vector<StorCell>> edge_pages_;
    int sz_page, data_bytes, idx_bytes;
};

#endif  // DRAMSIM3_GRAPH_H
