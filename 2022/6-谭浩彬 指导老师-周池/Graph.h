#include <bits/stdc++.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

#ifndef __GRAPH__
#define __GRAPH__

extern int THETA;             // 参数，决定一个顶点是否为High Degree
extern double DATA_UNIT;      // 参数，数据通信所消耗的基本数据流量大小，单位MB
extern double MOVE_DATA_UNIT; // 参数，移动一个顶点到其他服务器上消耗的流量，单位MB
typedef long long id_type;    // 决定顶点id的数据类型

class Vertex //顶点类
{
    static id_type id_count;

public:
    id_type id = -1;     // 顶点的唯一编号
    int current_dc = -1; // 当前所在的服务器
    int init_dc = -1;    // 初始所在的服务器
    int in_degree = 0;   // 顶点的入度
    int out_degree = 0;  // 顶点的出度

    int local_in_degree = 0;  // 顶点的本地入度
    int local_out_degree = 0; // 顶点的本地出度

    list<id_type> out_edge; // 顶点的出边邻居
    list<id_type> in_edge;  // 顶点的入边邻居
    // vector<id_type> out_edge;      // 顶点的出边邻居
    // vector<id_type> in_edge;       // 顶点的入边邻居
    bool is_high_degree = false; // 顶点是否为Hidh Degree
    int num_mirror;

    Vertex() // 初始化顶点的编号
    {
        id = id_count++;
    }
};

id_type Vertex::id_count = 0;

class Mirror // 镜像类
{
    static id_type id_count;

public:
    id_type id;               // 镜像的唯一id
    int local_in_degree = 0;  // 镜像的本地入度
    int local_out_degree = 0; // 镜像的本地出度
    bool in_use = false;      // 镜像是否在使用

    Mirror()
    {
        id = id_count++;
    }
    void del() // 删除镜像
    {
        local_in_degree = 0;local_out_degree = 0;
        in_use = false;
    }
    void add(int in, int out) //使用镜像
    {
        in_use = true;
        local_in_degree += in;
        local_out_degree += out;
    }
};

id_type Mirror::id_count = 0;

class DataCenter // 服务器类
{
public:
    double UploadBandwidth = -1;   // 上传带宽
    double DownloadBandwidth = -1; // 下载带宽
    double UploadPrice = -1;       // 上传价格 ($/MB)

    double gather_upload_wan = 0;   // 当前服务器在gather阶段上传使用的数据量
    double gather_download_wan = 0; // 当前服务器在gather阶段下载使用的数据量
    double apply_upload_wan = 0;    // 当前服务器在apply阶段上传使用的数据量
    double apply_download_wan = 0;  // 当前服务器在apply阶段下载使用的数据量

    long long vertex_num = 0; // 当前服务器在本地的顶点数量
    DataCenter(double up, double down, double price) : UploadBandwidth(up), DownloadBandwidth(down), UploadPrice(price){};
    void reset() // 重置服务器
    {
        gather_upload_wan = 0;
        gather_download_wan = 0;
        apply_upload_wan = 0;
        apply_download_wan = 0;
    }
};

class Graph // 图类
{
public:
    long long vertex_num = 0; // 顶点数量
    long long edge_num = 0;   // 边数量
    int DC_num = 0;           // 地理式分布的服务器数量

    double transfer_time; // 完成一次通信的时间
    double transfer_cost; // 完成一次通信消耗的价格
    double movecost;      // 移动顶点消耗的价格

    vector<Vertex> vertex;         // 所有的顶点
    vector<DataCenter> DC;         // 所有的服务器
    vector<vector<Mirror>> mirror; // 所有的镜像，服务器数量×顶点数量
    string GRAPH_FILE_NAME;        // 图文件名
    string NETWORK_FILE_NAME;      // 服务器文件名

    unordered_map<id_type, id_type> mapped; // 顶点映射

    unsigned long long file_lines; // 图文件长度
    unsigned long long high_degree_node_num = 0;

    Graph(string graph_file_name, string network_file_name)
    {
        GRAPH_FILE_NAME = graph_file_name;
        NETWORK_FILE_NAME = network_file_name;
    }

    void read_graph_file_vertex() // 读取图文件
    {
        ifstream file;
        file.open(GRAPH_FILE_NAME, ios::in);

        if (!file.is_open())
        {
            cout << "Can't not open graph file : " << GRAPH_FILE_NAME << endl;
            exit(-1);
        }
        file_lines = 0;
        stringstream ss;
        // ss << file.rdbuf();
        string s;

        id_type max_id = -1;
        id_type tmp;

        id_type mapped_count = 0;

        cout << "Finding max vertex id...\r" << endl;

        // 顶点的连续映射
        while (getline(file, s))
        {
            file_lines++;
            ss.clear();
            ss << s;
            ss >> tmp;
            if (!mapped.count(tmp))
                mapped[tmp] = mapped_count++;
            max_id = max(mapped[tmp], max_id);
            ss >> tmp;

            if (!mapped.count(tmp))
                mapped[tmp] = mapped_count++;
            max_id = max(mapped[tmp], max_id);
            ss.str("");
        }

        // vertex_num = max_id + 1;
        vertex_num = mapped.size();

        // 输出图顶点的数据情况
        cout << "[Graph State] max vertex id : " << max_id << endl;
        cout << "[Graph State] real vertex num : " << mapped.size() << endl;

        vertex = vector<Vertex>(vertex_num);
    }

    void read_graph_file_edge() // 读取图边的数据
    {
        ifstream file;
        file.open(GRAPH_FILE_NAME, ios::in);

        if (!file.is_open())
        {
            cout << "Can't not open graph file : " << GRAPH_FILE_NAME << endl;
            exit(-1);
        }

        stringstream ss;
        // ss << file.rdbuf();
        string s;

        id_type max_id = -1;
        id_type source, dest;
        id_type cur_line = 0;
        // 一个线程负责读取边，一个线程负责输出当前读取情况到屏幕
#pragma omp parallel num_threads(2) shared(cur_line)
        {

            if (omp_get_thread_num() == 0)
            {
                while (getline(file, s))
                {
                    ss.clear();
                    ss << s;
                    ss >> source >> dest;

                    source = mapped[source];
                    dest = mapped[dest];

                    edge_num++;

                    vertex[source].out_edge.push_back(dest);
                    vertex[source].out_degree++;

                    vertex[dest].in_edge.push_back(source);
                    vertex[dest].in_degree++;

                    if (vertex[dest].in_degree == THETA)
                        vertex[dest].is_high_degree = true, high_degree_node_num++;
                    ss.str("");

                    cur_line++;
                }
                // cur = vertex_num;
            }
            else
            {

                while (cur_line != file_lines)
                {
                    int schedule = 100 * cur_line / file_lines + 1;
                    printf("Loading edge : %d%% \t[", schedule);
                    for (int i = 0; i < 100; i++)
                    {
                        if (i < schedule)
                            cout << "*";
                        else
                            cout << " ";
                    }
                    cout << "]";
                    fflush(stdout);
                    usleep(10000);
                    cout << '\r';

                    // for (int i = 0; i < 200; i++)
                    //     cout << "\b";
                    fflush(stdout);
                }
                cout << endl
                     << "[Graph State] edge num : " << edge_num << endl;
            }
        }
    }

    void read_network_file() // 读取服务器文件内容
    {
        ifstream file;
        file.open(NETWORK_FILE_NAME, ios::in);

        if (!file.is_open())
        {
            cout << "Can't not open network file : " << NETWORK_FILE_NAME << endl;
            exit(-1);
        }

        stringstream ss;
        // ss << file.rdbuf();
        string s;

        double UploadBandwidth = -1;
        double DownloadBandwidth = -1;
        double UploadPrice = -1;

        while (getline(file, s))
        {
            ss.clear();
            ss << s;
            ss >> UploadBandwidth >> DownloadBandwidth >> UploadPrice;
            DC_num++;
            DC.emplace_back(DataCenter(UploadBandwidth, DownloadBandwidth, UploadPrice));
            ss.str("");
        }
        cout << "[Graph State] DC num : " << DC_num << endl;
    }

    void read_file() // 读取所有文件
    {
        read_graph_file_vertex();
        read_graph_file_edge();
        read_network_file();

        // 建立镜像
        vector<Mirror> m(vertex_num);
        for (int i = 0; i < DC_num; i++)
            mirror.push_back(m);
    }

    void hash_partition() // 对顶点按照id进行hash方式进行初始化分布
    {
#pragma omp parallel for
        for (int i = 0; i < vertex_num; i++)
        {
            // cout << omp_get_num_threads() << endl;
            int dc = i % DC_num;
            vertex[i].current_dc = vertex[i].init_dc = dc;
#pragma omp atomic
            DC[dc].vertex_num++;
        }
    }
    void average_partition() // 对顶点进行连续id整齐划分
    {
        int count = vertex_num / DC_num;
#pragma omp parallel for
        for (int i = 0; i < vertex_num; i++)
        {
            // cout << omp_get_num_threads() << endl;
            int dc = min(i / count, DC_num - 1);
            vertex[i].current_dc = vertex[i].init_dc = dc;
#pragma omp atomic
            DC[dc].vertex_num++;
        }
    }
    void random_partition() // 对顶点进行随机分布
    {
#pragma omp parallel for
        for (int i = 0; i < vertex_num; i++)
        {
            // cout << omp_get_num_threads() << endl;
            int dc = rand() % DC_num;
            vertex[i].current_dc = vertex[i].init_dc = dc;
#pragma omp atomic
            DC[dc].vertex_num++;
        }
    }
    void print_cross_dc_edge_num() // 输出跨dc的边的数量
    {
        long long cross_dc_edge_num = 0;
#pragma omp parallel for
        for (int i = 0; i < vertex_num; i++)
        {
            // cout << omp_get_num_threads() << endl;
            int dc = vertex[i].current_dc;
            for (auto x : vertex[i].in_edge)
                if (dc != vertex[x].current_dc)
#pragma omp atomic
                    cross_dc_edge_num++;
        }
        printf("Cross DC edge num : %lld (%f of total)\n", cross_dc_edge_num, 1. * cross_dc_edge_num / edge_num);
    }

    void hybrid_cut() // 对顶点进行hybrid cut模型的初始化
    {
        cout << "Begin hybrid cut" << endl;
        for (int i = 0; i < vertex_num; i++)
        {
            int vertex_dc = vertex[i].current_dc;

            if (vertex[i].is_high_degree) // High Degree的顶点将镜像添加到入边的邻顶上
            {
                for (auto x : vertex[i].in_edge)
                {
                    int edge_dc = vertex[x].current_dc;
                    vertex[x].local_out_degree++;

                    if (vertex_dc != edge_dc)
                        mirror[edge_dc][i].add(1, 0);
                    else
                        vertex[i].local_in_degree++;
                }
            }
            else // Low Degree的顶点将入边的邻顶镜像添加到自己服务器上
            {
                for (auto x : vertex[i].in_edge)
                {
                    int edge_dc = vertex[x].current_dc;
                    if (vertex_dc != edge_dc)
                        mirror[vertex_dc][x].add(0, 1);
                    else
                        vertex[x].local_out_degree++;

                    vertex[i].local_in_degree++;
                }
            }
        }
        movecost = 0;
#pragma omp parallel for
        for (int i = 0; i < vertex_num; i++)
        {
            for (int j = 0; j < DC_num; j++)
                vertex[i].num_mirror += mirror[j][i].in_use ? 1 : 0;
        }
    }
    void vertex_cut_example()
    {
        cout << "Begin vertex cut" << endl;
        for (int i = 0; i < vertex_num; i++)
        {
            int vertex_dc = vertex[i].current_dc;

            for (auto x : vertex[i].in_edge)
            {
                int edge_dc = vertex[x].current_dc;
                vertex[x].local_out_degree++;

                if (vertex_dc != edge_dc)
                {
                    mirror[edge_dc][i].add(1, 0), vertex[i].num_mirror++;
                    mirror[vertex_dc][x].add(0, 1), vertex[x].num_mirror++;
                    vertex[i].local_in_degree++;
                }
                else
                    vertex[i].local_in_degree++;
            }
        }
        movecost = 0;
        calculate_network_wan();

        print();
        exit(-1);
    }
    void reset_DC()
    {
        for (int i = 0; i < DC_num; i++)
            DC[i].reset();
    }
    void calculate_network_wan() // 计算一次通信消耗的流量
    {
        reset_DC();
        for (int dc = 0; dc < DC_num; dc++)
        {
            for (int v = 0; v < vertex_num; v++)
            {
                if (mirror[dc][v].in_use)
                {
                    int master_dc = vertex[v].current_dc;
                    if (mirror[dc][v].local_in_degree > 0)
                        DC[dc].gather_upload_wan += DATA_UNIT,
                            DC[master_dc].gather_download_wan += DATA_UNIT;
                    if (mirror[dc][v].local_out_degree > 0)
                        DC[master_dc].apply_upload_wan += DATA_UNIT,
                            DC[dc].apply_download_wan += DATA_UNIT;
                }
            }
        }
        calculate_network_time_price(transfer_time, transfer_cost);
    }
    void calculate_network_time_price(double &t, double &p) //计算一次通信消耗的时间与价格
    {
        double max_gather_t = 0, max_apply_t = 0;
        double sum_p = 0;
        for (int i = 0; i < DC_num; i++)
        {
            max_gather_t = max(max_gather_t, DC[i].gather_upload_wan / DC[i].UploadBandwidth);
            max_gather_t = max(max_gather_t, DC[i].gather_download_wan / DC[i].DownloadBandwidth);

            max_apply_t = max(max_apply_t, DC[i].apply_upload_wan / DC[i].UploadBandwidth);
            max_apply_t = max(max_apply_t, DC[i].apply_download_wan / DC[i].DownloadBandwidth);

            sum_p += DC[i].UploadPrice * (DC[i].apply_upload_wan + DC[i].gather_upload_wan);
        }
        t = max_apply_t + max_gather_t;
        p = sum_p;
    }
    void calculate_network_time_price(double &t, double &p, vector<DataCenter> &DC_)
    {
        double max_gather_t = 0, max_apply_t = 0;
        double sum_p = 0;
        for (int i = 0; i < DC_num; i++)
        {
            max_gather_t = max(max_gather_t, DC_[i].gather_upload_wan / DC_[i].UploadBandwidth);
            max_gather_t = max(max_gather_t, DC_[i].gather_download_wan / DC_[i].DownloadBandwidth);

            max_apply_t = max(max_apply_t, DC_[i].apply_upload_wan / DC_[i].UploadBandwidth);
            max_apply_t = max(max_apply_t, DC_[i].apply_download_wan / DC_[i].DownloadBandwidth);

            sum_p += DC_[i].UploadPrice * (DC_[i].apply_upload_wan + DC_[i].gather_upload_wan);
        }
        t = max_apply_t + max_gather_t;
        p = sum_p;
    }

    void print() // 打印状态信息
    {
        for (int i = 0; i < DC_num; i++)
        {
            printf("DC[%d] ---> vertex num: %lld GU: %f\tGD: %f\tAU: %f\tAD: %f\n", i, DC[i].vertex_num, DC[i].gather_upload_wan, DC[i].gather_download_wan, DC[i].apply_upload_wan, DC[i].apply_download_wan);
        }
        print_cross_dc_edge_num();
        double t, p;
        calculate_network_time_price(t, p);
        printf("Transfer time : %f\nTransfer cost : %f\tMove cost : %f\n", t, p, movecost);
    }
    void moveVertex(id_type v, int dc) // 移动顶点v到服务器dc
    {
        if (vertex[v].current_dc == dc)
            return;
        int origin_dc = vertex[v].current_dc;
        int init_dc = vertex[v].init_dc;
        if (origin_dc == init_dc) // 移出初始dc，需要产生消耗
        {
            movecost += MOVE_DATA_UNIT * DC[init_dc].UploadPrice;
        }
        else if (dc == init_dc) // 移回初始dc，取消消耗
        {
            movecost -= MOVE_DATA_UNIT * DC[init_dc].UploadPrice;
        }
        if (vertex[v].is_high_degree) // 移动的顶点是High Degree的
        {
            // 取消所有mirror的带来的消耗
            for (int i = 0; i < DC_num; i++)
            {
                if (mirror[i][v].in_use && mirror[i][v].local_in_degree > 0)
                    DC[origin_dc].gather_download_wan -= DATA_UNIT;
                if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                    DC[origin_dc].apply_upload_wan -= DATA_UNIT;
            }
            // 判断是否需要保留一个mirror
            if (vertex[v].local_in_degree > 0 || vertex[v].local_out_degree > 0)
            {
                mirror[origin_dc][v].add(vertex[v].local_in_degree, vertex[v].local_out_degree);
                vertex[v].num_mirror++;
                if (mirror[origin_dc][v].local_in_degree > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT;
                if (mirror[origin_dc][v].local_out_degree > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT;
            }
            else
                mirror[origin_dc][v].del();
            // 判断服务器dc是否已经有mirror
            if (mirror[dc][v].in_use)
            {
                if (mirror[dc][v].local_in_degree > 0)
                    DC[dc].gather_upload_wan -= DATA_UNIT;
                if (mirror[dc][v].local_out_degree > 0)
                    DC[dc].apply_download_wan -= DATA_UNIT;
                vertex[v].num_mirror--;
            }

            // 无论是否有镜像，先继承mirror已有的本地关系
            vertex[v].local_in_degree = mirror[dc][v].local_in_degree;
            vertex[v].local_out_degree = mirror[dc][v].local_out_degree;
            // mirror[dc][v].in_use = false;
            mirror[dc][v].del();

            // 更新mirror带来的消耗
            for (int i = 0; i < DC_num; i++)
            {
                if (mirror[i][v].in_use && mirror[i][v].local_in_degree > 0)
                    DC[dc].gather_download_wan += DATA_UNIT;
                if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                    DC[dc].apply_upload_wan += DATA_UNIT;
            }

            // 迁移出边
            for (auto out_neighbour : vertex[v].out_edge)
            {
                // 移动出边为High Degree的邻顶
                if (vertex[out_neighbour].is_high_degree)
                {
                    vertex[v].local_out_degree++;
                    mirror[origin_dc][v].local_out_degree--;

                    if (mirror[origin_dc][v].local_out_degree == 0)
                    {
                        DC[origin_dc].apply_download_wan -= DATA_UNIT;
                        DC[dc].apply_upload_wan -= DATA_UNIT;
                        if (mirror[origin_dc][v].local_in_degree == 0)
                            mirror[origin_dc][v].del(), vertex[v].num_mirror--;
                    }

                    if (vertex[out_neighbour].current_dc == origin_dc)
                    {
                        vertex[out_neighbour].local_in_degree--;

                        if (!mirror[dc][out_neighbour].in_use)
                            vertex[out_neighbour].num_mirror++;

                        if (mirror[dc][out_neighbour].local_in_degree == 0)
                        {
                            DC[dc].gather_upload_wan += DATA_UNIT;
                            DC[origin_dc].gather_download_wan += DATA_UNIT;
                        }
                        mirror[dc][out_neighbour].add(1, 0);
                    }
                    else
                    {
                        int out_neighbour_dc = vertex[out_neighbour].current_dc;
                        mirror[origin_dc][out_neighbour].local_in_degree--;
                        if (mirror[origin_dc][out_neighbour].local_in_degree == 0)
                        {
                            DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                            DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            if (mirror[origin_dc][out_neighbour].local_out_degree == 0)
                                mirror[origin_dc][out_neighbour].del(), vertex[out_neighbour].num_mirror--;
                        }
                        if (out_neighbour_dc == dc)
                        {
                            vertex[out_neighbour].local_in_degree++;
                        }
                        else
                        {
                            if (!mirror[dc][out_neighbour].in_use)
                                vertex[out_neighbour].num_mirror++;
                            if (mirror[dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[dc].gather_upload_wan += DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                            }
                            mirror[dc][out_neighbour].add(1, 0);
                        }
                    }
                }
            }
        }
        else // 移动的顶点是Low Degree的
        {
            // 首先移出mirror带来的消耗
            for (int i = 0; i < DC_num; i++)
            {
                if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                    DC[origin_dc].apply_upload_wan -= DATA_UNIT;
            }
            // 判断是否需要保留mirror
            if (vertex[v].local_in_degree > 0 || vertex[v].local_out_degree > 0)
            {
                mirror[origin_dc][v].add(vertex[v].local_in_degree, vertex[v].local_out_degree);
                vertex[v].num_mirror++;
                if (vertex[v].local_in_degree > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT,
                        DC[dc].gather_download_wan += DATA_UNIT; //需要临时加上
                if (vertex[v].local_out_degree > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT;
            }
            else
                mirror[origin_dc][v].del();
            // 判断是否已有mirror
            if (mirror[dc][v].in_use)
            {
                if (mirror[dc][v].local_in_degree > 0)
                    DC[dc].gather_upload_wan -= DATA_UNIT;
                if (mirror[dc][v].local_out_degree > 0)
                    DC[dc].apply_download_wan -= DATA_UNIT;
                vertex[v].num_mirror--;
            }
            // 继承mirror的关系
            vertex[v].local_in_degree = mirror[dc][v].local_in_degree;
            vertex[v].local_out_degree = mirror[dc][v].local_out_degree;
            // mirror[dc][v].in_use = false;
            mirror[dc][v].del();

            // 恢复mirror的消耗
            for (int i = 0; i < DC_num; i++)
            {
                if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                    DC[dc].apply_upload_wan += DATA_UNIT;
            }

            // 移动入边
            for (auto in_neighbour : vertex[v].in_edge)
            {
                mirror[origin_dc][v].local_in_degree--;
                vertex[v].local_in_degree++;

                if (mirror[origin_dc][v].local_in_degree == 0)
                {
                    DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                    DC[dc].gather_download_wan -= DATA_UNIT;
                    if (mirror[origin_dc][v].local_out_degree == 0)
                        mirror[origin_dc][v].del(), vertex[v].num_mirror--;
                }
                // if (vertex[in_neighbour].is_high_degree)
                {
                    int in_neighbour_dc = vertex[in_neighbour].current_dc;
                    if (in_neighbour_dc == origin_dc)
                    {
                        vertex[in_neighbour].local_out_degree--;
                        if (!mirror[dc][in_neighbour].in_use)
                            vertex[in_neighbour].num_mirror++;
                        if (mirror[dc][in_neighbour].local_out_degree == 0)
                        {
                            DC[dc].apply_download_wan += DATA_UNIT;
                            DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                        }
                        mirror[dc][in_neighbour].add(0, 1);
                    }
                    else
                    {
                        mirror[origin_dc][in_neighbour].local_out_degree--;
                        if (mirror[origin_dc][in_neighbour].local_out_degree == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[in_neighbour_dc].apply_upload_wan -= DATA_UNIT;
                            if (mirror[origin_dc][in_neighbour].local_in_degree == 0)
                                mirror[origin_dc][in_neighbour].del(), vertex[in_neighbour].num_mirror--;
                        }
                        if (in_neighbour_dc == dc)
                        {
                            vertex[in_neighbour].local_out_degree++;
                        }
                        else
                        {
                            if (!mirror[dc][in_neighbour].in_use)
                                vertex[in_neighbour].num_mirror++;
                            if (mirror[dc][in_neighbour].local_out_degree == 0)
                            {
                                DC[dc].apply_download_wan += DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                            }
                            mirror[dc][in_neighbour].add(0, 1);
                        }
                    }
                }
            }

            // 移动High Degree的出边
            for (auto out_neighbour : vertex[v].out_edge)
            {
                if (vertex[out_neighbour].is_high_degree)
                {
                    vertex[v].local_out_degree++;
                    mirror[origin_dc][v].local_out_degree--;

                    if (mirror[origin_dc][v].local_out_degree == 0)
                    {
                        DC[origin_dc].apply_download_wan -= DATA_UNIT;
                        DC[dc].apply_upload_wan -= DATA_UNIT;
                        mirror[origin_dc][v].del();
                        vertex[v].num_mirror--;
                    }

                    if (vertex[out_neighbour].current_dc == origin_dc)
                    {
                        vertex[out_neighbour].local_in_degree--;
                        if (!mirror[dc][out_neighbour].in_use)
                            vertex[out_neighbour].num_mirror++;

                        if (mirror[dc][out_neighbour].local_in_degree == 0)
                        {
                            DC[dc].gather_upload_wan += DATA_UNIT;
                            DC[origin_dc].gather_download_wan += DATA_UNIT;
                        }
                        mirror[dc][out_neighbour].add(1, 0);
                    }
                    else
                    {
                        int out_neighbour_dc = vertex[out_neighbour].current_dc;
                        mirror[origin_dc][out_neighbour].local_in_degree--;
                        if (mirror[origin_dc][out_neighbour].local_in_degree == 0)
                        {
                            DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                            DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            if (mirror[origin_dc][out_neighbour].local_out_degree == 0)
                                mirror[origin_dc][out_neighbour].del(), vertex[out_neighbour].num_mirror--;
                        }
                        if (out_neighbour_dc == dc)
                        {
                            vertex[out_neighbour].local_in_degree++;
                        }
                        else
                        {
                            if (!mirror[dc][out_neighbour].in_use)
                                vertex[out_neighbour].num_mirror++;
                            if (mirror[dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[dc].gather_upload_wan += DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                            }
                            mirror[dc][out_neighbour].add(1, 0);
                        }
                    }
                }
            }
        }
        DC[origin_dc].vertex_num--;
        DC[dc].vertex_num++;
        vertex[v].current_dc = dc;

        calculate_network_time_price(transfer_time, transfer_cost);
    }

    void test()
    {
        int count = vertex_num / DC_num;
        for (int i = 0; i < vertex_num; i++)
            //     if (!vertex[i].is_high_degree)
            moveVertex(i, min(i / count, DC_num - 1));
        for (int i = 0; i < vertex_num; i++)
            //     // if (!vertex[i].is_high_degree)
            moveVertex(i, i % DC_num);
    }
    void output()
    {
        ofstream file;
        file.open("sb.txt", ios::out | ios::trunc);
        for (auto &v : vertex)
        {
            file << v.id << '\t' << v.current_dc << endl;
        }
        file.close();
        cout << "Finished file output" << endl;
    }
};

#endif