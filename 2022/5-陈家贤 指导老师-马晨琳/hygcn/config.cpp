//
// Created by szu on 2021/11/15.
//

#include <cassert>
#include <complex>
#include "INIReader.h"
#include "config.h"

void HyConfig::Init(const string &graph_path, const string &graph_name, const string &sys_path, const string &layer_path) {
    loader = std::make_shared<GraphLoader>();
    sys_config = std::make_shared<SysConfig>(sys_path);
    layer_config = std::make_shared<LayerConfig>(layer_path);
    loader->LoadGraph(graph_path, graph_name);
    if(layer_config->model_name == "gcn" || layer_config->model_name == "gin") {
        loader->raw_graph = loader->GetGcnNormGraph();
    } else if(layer_config->model_name == "gs") {
        loader->raw_graph = loader->LoadSampleGraph(layer_config->sample); //loader->GetSampleGraph(layer_config->sample);
    } else {
        dramsim3::AbruptExit(__FILE__, __LINE__);
    }
    PrepareNextLayer();
}

SysConfig::SysConfig(const std::string& config_path) {
    INIReader sys_reader(config_path);

    block_size = sys_reader.GetInteger("sys", "block_size", -1);
    cap_hbm = sys_reader.GetInteger("sys", "cap_hbm", -1);
    cap_hbm *= 1024 * 1024 * 1024;
    assert(block_size > 0 && cap_hbm > 0);

    num_simd = sys_reader.GetInteger("cpt", "num_simd", -1);
    num_core_per_simd = sys_reader.GetInteger("cpt", "num_core_per_simd", -1);
    systolic_h = sys_reader.GetInteger("cpt", "systolic_h", -1);
    systolic_w = sys_reader.GetInteger("cpt", "systolic_w", -1);
    assert(num_simd > 0 && num_core_per_simd > 0 && systolic_h > 0 && systolic_w > 0);

    cap_inbuf = sys_reader.GetInteger("buffer", "cap_inbuf", -1);
    cap_edgebuf = sys_reader.GetInteger("buffer", "cap_edgebuf", -1);
    cap_wbuf = sys_reader.GetInteger("buffer", "cap_wbuf", -1);
    cap_outbuf = sys_reader.GetInteger("buffer", "cap_outbuf", -1);
    cap_aggrbuf = sys_reader.GetInteger("buffer", "cap_aggrbuf", -1);

    cap_comb_inbuf = sys_reader.GetInteger("buffer", "cap_comb_inbuf", -1);
    assert(cap_comb_inbuf > 0);
    assert(cap_inbuf > 0 && cap_edgebuf > 0 && cap_wbuf > 0 && cap_outbuf > 0 && cap_aggrbuf > 0);




}


LayerConfig::LayerConfig(const std::string& layer_path) {
    layer_reader = std::make_shared<INIReader>(layer_path);
    layer_count = -1;
    num_layer = layer_reader->GetInteger("gcn", "num_layer", -1);
    data_bytes = layer_reader->GetInteger("gcn", "data_bytes", -1);
    idx_bytes = layer_reader->GetInteger("gcn", "idx_bytes", -1);
    model_name = layer_reader->Get("gcn", "model", "nmsl");
    sample = layer_reader->GetInteger("gcn", "sample", -1);
    assert(model_name != "nmsl");
    if(model_name == "gs" && sample <= 0) {
        dramsim3::AbruptExit(__FILE__, __LINE__);
    }

    assert(num_layer > 0 && data_bytes > 0 && idx_bytes > 0);


}

bool HyConfig::PrepareNextLayer() {
    if (layer_config->layer_count == layer_config->num_layer - 1) {
        return false;
    }

    layer_config->layer_count++;

    assert(layer_config->num_layer == 2);


    std::string layer_name = std::string("layer") + std::to_string(layer_config->layer_count);
    layer_config->weight_h = layer_config->layer_reader->GetInteger(layer_name, "weight_h", -1);
    layer_config->weight_w = layer_config->layer_reader->GetInteger(layer_name, "weight_w", -1);

    assert(layer_config->weight_h > 0 && layer_config->weight_w > 0);

    if(layer_config->layer_count == 0) {
        layer_config->weight_h = loader->raw_graph.len_feature;
        if(layer_config->model_name == "gin") {
            layer_config->weight_w = (double)(layer_config->weight_h + 128) / layer_config->weight_h * 128;
        }
    } else if(layer_config->layer_count == layer_config->num_layer - 1) {
        layer_config->weight_w = loader->raw_graph.num_class;
        if(layer_config->model_name == "gin") {
            layer_config->weight_w = (double)(layer_config->weight_h + 128) / layer_config->weight_h * 128;
        }
    }

    tile_config = std::make_shared<TileConfig>(*this);
    mapper = std::make_shared<AddrMapper>(loader, sys_config, layer_config, tile_config);

    return true;

}



AddrMapper::AddrMapper (const std::shared_ptr<GraphLoader> &loader,
            const std::shared_ptr<SysConfig> &sys_config,
            const std::shared_ptr<LayerConfig> &layer_config,
            const std::shared_ptr<TileConfig> &tile_config):
        sz_block(sys_config->block_size),
        num_vertex(loader->raw_graph.num_vertex),
        space_edge_chunk(sys_config->cap_edgebuf),
        idx_bytes(layer_config->idx_bytes),
        data_bytes(layer_config->data_bytes),
        tile_config(tile_config)
{

    hbm_start_addr = 0;
    hbm_end_addr = sys_config->cap_hbm;

    input_start_addr = hbm_start_addr;
    input_end_addr = input_start_addr + num_vertex * layer_config->space_aligned_vertex(sz_block);

    output_start_addr = input_end_addr;
    output_end_addr = output_start_addr + num_vertex * layer_config->space_aligned_output_vertex(sz_block);

    weight_start_addr = output_end_addr;
    weight_end_addr = output_end_addr + tile_config->space_aligned_weight;
    assert(space_edge_chunk % sz_block == 0);

    int max_vertex = tile_config->ele_aggr_w_interval;
    int edge_size = 0;
    int vertex_size = 0;
    for(int i = 0; i < num_vertex;) {
        int cur_edge_info = 0;
        cur_edge_info += layer_config->idx_bytes;
        cur_edge_info += loader->raw_graph.r_adj[i].size() * (idx_bytes + data_bytes);
        if(cur_edge_info > space_edge_chunk) {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }
        if(edge_size + cur_edge_info > space_edge_chunk || vertex_size > max_vertex - 1) {
            auto start_iter = loader->raw_graph.r_adj.begin();
            if(!end_vertex_of_echunk.empty()) {
                start_iter += end_vertex_of_echunk.back();
            }
            auto end_iter = loader->raw_graph.r_adj.begin() + i;
            edge_chunk.push_back(std::vector<std::vector<int>>(start_iter, end_iter));
            end_vertex_of_echunk.push_back(i);
            edge_size = 0;
            vertex_size = 0;
        } else {
            edge_size += cur_edge_info;
            vertex_size++;
            i++;
        }
    }

    if(edge_size != 0) {
        auto start_iter = loader->raw_graph.r_adj.begin();
        if(!end_vertex_of_echunk.empty()) {
            start_iter += end_vertex_of_echunk.back();
        }
        auto end_iter = loader->raw_graph.r_adj.begin() + num_vertex;
        edge_chunk.push_back(std::vector<std::vector<int>>(start_iter, end_iter));
        end_vertex_of_echunk.push_back(num_vertex);
    }

    assert(edge_chunk.size() == end_vertex_of_echunk.size());
    num_edge_chunk = edge_chunk.size();

    start_vertex_of_echunk.reserve(num_edge_chunk);
    start_vertex_of_echunk.push_back(0);
    for(int i = 0; i < num_edge_chunk - 1; i++) {
        start_vertex_of_echunk.push_back(end_vertex_of_echunk[i]);
    }

    edge_start_addr = weight_end_addr;
    edge_end_addr = edge_start_addr + num_edge_chunk * space_edge_chunk;

    return;

}

TileConfig::TileConfig(const HyConfig& hy_config) {

    const std::shared_ptr<SysConfig>& sys_config = hy_config.sys_config;
    const std::shared_ptr<LayerConfig>& layer_config = hy_config.layer_config;
    int data_bytes = layer_config->data_bytes;
    int sz_block = sys_config->block_size;
    int ele_block = sys_config->block_size / data_bytes;
    this->aligned_weight_h = UpperAlign(layer_config->weight_h, ele_block);
    this->aligned_weight_w = UpperAlign(layer_config->weight_w, ele_block);
    int space_weights =  this->aligned_weight_h * this->aligned_weight_w * data_bytes;
    space_aligned_weight = space_weights;


    int cap_wbuf = sys_config->cap_wbuf;
    if(space_weights <= sys_config->cap_wbuf) {
        this->ele_comb_w_chunk = aligned_weight_w;
        this->ele_comb_h_chunk = aligned_weight_h;
    } else {

        int ele_chunk = LowerAlign(std::sqrt(cap_wbuf / data_bytes), ele_block);
        if (aligned_weight_h >= ele_chunk && aligned_weight_w >= ele_chunk) {
            this->ele_comb_w_chunk = this->ele_comb_h_chunk = ele_chunk;

        } else if (aligned_weight_h <= ele_chunk && aligned_weight_w > ele_chunk){
            this->ele_comb_h_chunk = this->aligned_weight_h;
            double ele_w_chunk = cap_wbuf / data_bytes / this->ele_comb_h_chunk;
            this->ele_comb_w_chunk = LowerAlign(ele_w_chunk , ele_block);

        } else if (aligned_weight_w <= ele_chunk && aligned_weight_h > ele_chunk) {
            this->ele_comb_w_chunk = this->aligned_weight_w;
            double ele_h_chunk = cap_wbuf / data_bytes / this->ele_comb_w_chunk;
            this->ele_comb_h_chunk = LowerAlign(ele_h_chunk, ele_block);
        } else {
            dramsim3::AbruptExit(__FILE__, __LINE__);
        }

    }

    this->sz_comb_h_chunk = this->ele_comb_h_chunk * data_bytes;
    this->sz_comb_w_chunk = this->ele_comb_w_chunk * data_bytes;

    this->num_comb_h_chunk = std::ceil((double)this->aligned_weight_h / this->ele_comb_h_chunk);
    this->num_comb_w_chunk = std::ceil((double)this->aligned_weight_w / this->ele_comb_w_chunk);


    int space_vector = layer_config->space_aligned_vertex(sz_block);
    this->ele_aggr_h_interval = sys_config->cap_inbuf / space_vector;
    //
    int num_half_vertex = std::ceil((double)hy_config.loader->raw_graph.num_vertex / 2);
    this->ele_aggr_w_interval = sys_config->cap_aggrbuf / space_vector;
    this->ele_aggr_w_interval = std::min(this->ele_aggr_w_interval, num_half_vertex);


    this->sz_comb_batch = std::min(sys_config->cap_comb_inbuf / this->sz_comb_h_chunk, sys_config->cap_outbuf / this->sz_comb_w_chunk);
    if(this->sz_comb_batch <= 1) {
        dramsim3::AbruptExit(__FILE__, __LINE__);
    }

    space_aligned_weight = this->ele_comb_w_chunk * this->ele_comb_h_chunk * data_bytes
                           * num_comb_h_chunk * num_comb_w_chunk;

    std::cout << "store amplification:" << (double) space_aligned_weight / (layer_config->weight_h * layer_config->weight_w * data_bytes) << std::endl;
    std::cout << "weight_buf utilization:" << (double)(ele_comb_h_chunk * ele_comb_w_chunk * data_bytes) / sys_config->cap_wbuf << std::endl;
    std::cout << "input_buf utilization:" << (double)(sz_comb_batch * sz_comb_h_chunk) / sys_config->cap_comb_inbuf << std::endl;
    std::cout << "output_buf utilization:" << (double)(sz_comb_batch * sz_comb_w_chunk) / sys_config->cap_outbuf << std::endl;


    space_weight_block = ele_comb_h_chunk * ele_comb_w_chunk * data_bytes;

    return;


}