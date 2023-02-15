#include <bits/stdc++.h>
#include "Graph.h"
#include "LearningAutomaton.h"

using namespace std;

string GIRAPH_FILE_NAME;
int THETA = 100;
double DATA_UNIT = 0.000008;
double MOVE_DATA_UNIT = 10;
double train_rate = 1.0;
double sampling_rate = 0.2;
double Budget_rate = 0.2;
bool auto_sampling_rate = false;
double overhead_limit;
int block_nums = 0;

int main(int argc, char *argv[])
{
    string graph_file_name;
    string network_file_name;
    enum ALGORITHM
    {
        LA,              //最基本的LA算法（无惩罚）
        LA_with_Penalty, //带有惩罚的算法
        LA_with_Cancel,     //带撤销操作
        LA_with_Cancel_with_Greedy, //贪心搜索
        LA_with_Cancel_with_Greedy_HighDegreeNode,  //只训练高度数节点
        LA_with_Cancel_with_Greedy_LowDegreeNode,   //只训练低度数节点
        LA_with_Cancel_with_Greedy_MDegreeNode,     //只训练度数小于等于M的节点
        LA_with_Cancel_with_Greedy_TOPKDegreeNode,   //只训练TOP-K的最小度数节点
        LA_with_Cancel_with_Greedy_Sampling,
        LA_with_Cancel_with_Greedy_Block,
        LA_with_Cancel_with_Greedy_Mirror_Seq
    };
    int algorithm = -1;
    int iteration = 20;

    int o;
    const char *optstring = "g:n:a:t:k:s:r:o:b:";

    while ((o = getopt(argc, argv, optstring)) != -1)
    {
        switch (o)
        {
        case 'g':   //训练的图文件名称
            graph_file_name = optarg;
            printf("[graph_file_name]: %s\n", graph_file_name.c_str());
            break;
        case 'n':   //训练的网络文件名称
            network_file_name = optarg;
            printf("[network_file_name]: %s\n", network_file_name.c_str());
            break;
        case 'a':   //训练的算法
            if (strcmp("LA", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA");
                algorithm = LA;
            }
            else if (strcmp("LA_with_Penalty", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Penalty");
                algorithm = LA_with_Penalty;
            }
            else if (strcmp("LA_with_Cancel", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel");
                algorithm = LA_with_Cancel;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy");
                algorithm = LA_with_Cancel_with_Greedy;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_LowDegreeNode", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_LowDegreeNode");
                algorithm = LA_with_Cancel_with_Greedy_LowDegreeNode;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_HighDegreeNode", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_HighDegreeNode");
                algorithm = LA_with_Cancel_with_Greedy_HighDegreeNode;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_MDegreeNode", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_MDegreeNode");
                algorithm = LA_with_Cancel_with_Greedy_MDegreeNode;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_TOPKDegreeNode", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_TOPKDegreeNode");
                algorithm = LA_with_Cancel_with_Greedy_TOPKDegreeNode;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_Sampling", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_Sampling");
                algorithm = LA_with_Cancel_with_Greedy_Sampling;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_Block", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_Block");
                algorithm = LA_with_Cancel_with_Greedy_Block;
            }
            else if (strcmp("LA_with_Cancel_with_Greedy_Mirror_Seq", optarg) == 0)
            {
                printf("[algorithm]: %s\n", "LA_with_Cancel_with_Greedy_Mirror_Seq");
                algorithm = LA_with_Cancel_with_Greedy_Mirror_Seq;
            }
            else
            {
                printf("[algorithm]: ERROR!!!(%s)\n", optarg);
                exit(-1);
            }

            break;
        case 't':   //迭代次数
            iteration = atoi(optarg);
            printf("[iteration]: %d\n", iteration);
            break;
        case 'k':   //TOP K
            train_rate = atof(optarg);
            if (train_rate < 0 || train_rate > 1)
            {
                printf("[ERROR]: %f(train rate must in [0, 1])\n", train_rate);
                exit(-1);
            }
            printf("[train rate]: %f\n", train_rate);
            break;
        case 's':
        {
            double tmp_sampling_rate = atof(optarg);
            if(tmp_sampling_rate < 0 || tmp_sampling_rate > 1)
            {
                printf("[ERROR]: %f(sampling rate must in [0, 1])\n", train_rate);
                exit(-1);
            }
            sampling_rate = tmp_sampling_rate;
            printf("[sampling rate]: %f\n", sampling_rate);
            break;
        }
        case 'r':
        {
            double tmp_budget_rate = atof(optarg);
            if(tmp_budget_rate < 0 || tmp_budget_rate > 1)
            {
                printf("[ERROR]: %f(Budget rate must in [0, 1])\n", train_rate);
                exit(-1);
            }
            Budget_rate = tmp_budget_rate;
            printf("[Budget rate]: %f\n", Budget_rate);
            break;
        }
        case 'o':
        {
            overhead_limit = atof(optarg);
            printf("[overhead limit]: %f\n", overhead_limit);
            auto_sampling_rate = true;
            break;
        }
        case 'b':
        {
            block_nums = atoi(optarg);
            printf("[Block nums]: %d\n", block_nums);
            break;
        }

        case '?':
            printf("Unknown option: %c\n", (char)optopt);
            exit(-1);
            break;
        }
    }
    // if (argc != 9 && argc != 11)
    // {
    //     cout << "Insufficient parameters!!" << endl;
    //     exit(-1);
    // }
    // Graph *graph = new Graph("/home/local_graph/soc-LiveJournal1.txt", "/home/thb/GuduationDesign/network/Amazon.txt");
    // Graph *graph = new Graph("/home/cluster_share/graph/web-Google.txt", "/home/thb/GuduationDesign/network/Amazon.txt");
    // Graph *graph = new Graph("/home/cluster_share/graph/debug.txt", "/home/thb/GuduationDesign/network/Amazon.txt");
    Graph *graph = new Graph(graph_file_name, network_file_name);
    graph->read_file();
    // graph->average_partition();
    // graph->hash_partition();
    graph->random_partition();
    graph->hybrid_cut();
    // graph->print();

    graph->calculate_network_wan();

    graph->print();
    // graph->vertex_cut_example();
    // exit(-1);

    LearningAutomaton learningautomaton(graph, iteration);
    learningautomaton.init();

    switch (algorithm)
    {
    case LA:
        cout << "============ running LA ============" << endl;
        learningautomaton.baseline_LA();
        break;
    case LA_with_Penalty:
        cout << "============ running LA_with_Penalty ============" << endl;
        learningautomaton.baseline_LA_with_Penalty();
        break;
    case LA_with_Cancel:
        cout << "============ running LA_with_Cancel ============" << endl;
        learningautomaton.baseline_LA_with_Cancel();
        break;
    case LA_with_Cancel_with_Greedy:
        cout << "============ running LA_with_Cancel_with_Greedy ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy();
        break;
    case LA_with_Cancel_with_Greedy_HighDegreeNode:
        cout << "============ running LA_with_Cancel_with_Greedy_HighDegreeNode ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_HighDegreeNode();
        break;
    case LA_with_Cancel_with_Greedy_LowDegreeNode:
        cout << "============ running LA_with_Cancel_with_Greedy_LowDegreeNode ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_LowDegreeNode();
        break;
    case LA_with_Cancel_with_Greedy_MDegreeNode:
        cout << "============ running LA_with_Cancel_with_Greedy_MDegreeNode ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_MDegreeNode();
        break;
    case LA_with_Cancel_with_Greedy_TOPKDegreeNode:
        cout << "============ running LA_with_Cancel_with_Greedy_TOPKDegreeNode ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_TOPKDegreeNode();
        break;
    case LA_with_Cancel_with_Greedy_Sampling:
        cout << "============ running LA_with_Cancel_with_Greedy_Sampling ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_Sampling();
        break;
    case LA_with_Cancel_with_Greedy_Block:
        cout << "============ running LA_with_Cancel_with_Greedy_Block ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_Block();
        break;
    case LA_with_Cancel_with_Greedy_Mirror_Seq:
        cout << "============ running LA_with_Cancel_with_Greedy_Mirror_Seq ============" << endl;
        learningautomaton.baseline_LA_with_Cancel_with_Greedy_Mirror_Seq();
        break;

    default:
        break;
    }

    // LA.output();
}