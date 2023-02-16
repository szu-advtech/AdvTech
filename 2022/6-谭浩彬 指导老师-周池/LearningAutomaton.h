#include "Graph.h"
#include <bits/stdc++.h>
#include <omp.h>
#include <time.h>
#include <atomic>
#include <semaphore.h>

#ifndef __LA__
#define __LA__

extern double train_rate;
extern double Budget_rate;
extern bool auto_sampling_rate;
extern double overhead_limit;
extern double sampling_rate;
extern int block_nums;

class LearningAutomaton
{
    Graph *graph = NULL;
    vector<vector<double>> probability; // 概率
    // vector<int> choice_dc_record;
    int iteration = 50; // 迭代次数
    double Budget = 0;  // Budget预算
    double alpha = 0.2; // Alpha参数
    // double Budget_rate = 0.2; // Budget占的比例
    double origin_time; // 初始通信时间
    double origin_cost; // 初始开销
    int batch_size = 8;
    vector<int> action;
    vector<id_type> training_v;
    vector<id_type> pre_choice;

    vector<atomic_int> atomic_gather_upload_wan;
    vector<atomic_int> atomic_gather_download_wan;
    vector<atomic_int> atomic_apply_upload_wan;
    vector<atomic_int> atomic_apply_download_wan;

    vector<vector<pthread_mutex_t>> mirror_mutex;
    // pthread_spinlock_t **mirror_mutex;
    ofstream log_file;
    vector<id_type> id_vector;

public:
    LearningAutomaton(Graph *g, int it = 20)
    {
        graph = g;
        atomic_gather_upload_wan = vector<atomic_int>(g->DC_num);
        atomic_gather_download_wan = vector<atomic_int>(g->DC_num);
        atomic_apply_upload_wan = vector<atomic_int>(g->DC_num);
        atomic_apply_download_wan = vector<atomic_int>(g->DC_num);
        action.resize(batch_size);
        training_v.resize(batch_size);
        pre_choice.resize(graph->vertex_num);
        iteration = it;

        mirror_mutex = vector<vector<pthread_mutex_t>>(g->DC_num, vector<pthread_mutex_t>(graph->vertex_num));
        for (int i = 0; i < g->DC_num; i++)
            for (int j = 0; j < g->vertex_num; j++)
                pthread_mutex_init(&mirror_mutex[i][j], NULL);
        omp_set_nested(1);
        log_file.open("LOG.txt", ios::out | ios::trunc);
        if (!log_file.is_open())
        {
            cout << "can't open LOG file" << endl;
            exit(-1);
        }
    }
    ~LearningAutomaton()
    {
        log_file.close();
    }
    void output()
    {
        graph->output();
    }
    void init()
    {
        // 初始化概率，迁移到每个服务器都是均等的
        probability = vector<vector<double>>(graph->vertex_num, vector<double>(graph->DC_num, 1. / graph->DC_num));
        // choice_dc_record = vector<int>(graph->vertex_num);
        int max_price_dc = 0;                        // 记录最贵的服务器
        double max_price = graph->DC[0].UploadPrice; // 记录最贵的服务器上传价格
        Budget = 0;
        for (int i = 0; i < graph->DC_num; i++)
        {
            if (max_price < graph->DC[i].UploadPrice)
                max_price_dc = i, max_price = graph->DC[i].UploadPrice;
            Budget += graph->DC[i].vertex_num * graph->DC[i].UploadPrice;
        }
        Budget -= graph->DC[max_price_dc].vertex_num * graph->DC[max_price_dc].UploadPrice;
        Budget *= Budget_rate;
        Budget *= MOVE_DATA_UNIT; // Budget为将所有顶点迁移到最贵的DC所消耗的价格的budget_rate
        origin_time = graph->transfer_time;
        origin_cost = graph->transfer_cost;
    }
    inline double genRandom() // 获取0~1的随机数
    {
        return 1. * rand() / RAND_MAX;
    }
    int make_decision_greedy(id_type v)
    {
        int choice_dc = max_element(probability[v].begin(), probability[v].end()) - probability[v].begin();
        return choice_dc;
    }
    int make_decision_roulette(id_type v)
    {
        double r = genRandom();
        double sum_pro = 0;
        int choice_dc = rand() % graph->DC_num;
        // 轮盘赌算法
        for (int i = 0; i < graph->DC_num; i++)
        {
            sum_pro += probability[v][i];
            if (sum_pro >= r)
            {
                choice_dc = i;
                break;
            }
        }
        return choice_dc;
    }
    double Score(double old_time, double old_cost, double old_mvcost, double new_time, double new_cost, double new_mvcost, int iter)
    // 计算得分
    {
        double score;
        // if (new_cost + new_mvcost < Budget)
        score = (old_time - new_time) / old_time + (old_cost - new_cost) / old_cost;
        // else

        if (new_mvcost > Budget)
        {
            double old_ = old_cost + old_mvcost;
            double new_ = new_cost + new_mvcost;
            double b = 0.5;

            double rate = b * (1+iter) / iteration;

            // score = (1 - rate) * (old_time - new_time) / old_time + (0+rate) * (old_ - new_) / old_;
            // score = (0.5 - rate) * score +  (0.5 + rate) * (old_mvcost - new_mvcost) ;
            score = (b - rate) * score + (1 - b + rate) * (old_mvcost - new_mvcost) / old_mvcost;
        }

        /*
        if (new_mvcost > Budget && iter > iteration /  2)
         {
             double old_ = old_cost + old_mvcost;
             double new_ = new_cost + new_mvcost;
             double b = 0.1;

             double rate = b * iter / iteration;

             // score = (1 - rate) * (old_time - new_time) / old_time + (0+rate) * (old_ - new_) / old_;
             // score = (0.5 - rate) * score +  (0.5 + rate) * (old_mvcost - new_mvcost) ;
             score = score + (old_mvcost - new_mvcost) / old_mvcost;
         }
         */

        // if(old_cost - new_cost > 0)
        // cout << score << endl;
        // return -1;
        return score;
    }
    bool Signal(double old_time, double old_cost, double old_mvcost, double new_time, double new_cost, double new_mvcost, int iter)
    {
        return Score(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, iter) > 0;
    }

    void reset_virtual_DC()
    {
        for (int i = 0; i < graph->DC_num; i++)
        {
            atomic_apply_download_wan[i] = 0;
            atomic_apply_upload_wan[i] = 0;
            atomic_gather_download_wan[i] = 0;
            atomic_gather_upload_wan[i] = 0;
        }
    }

    vector<double> moveVirtualVertex(id_type v, int it)
    // 以当前环境为参考，移动顶点到所有dc，返回每个dc的score情况
    {
        int &DC_num = graph->DC_num;
        vector<Vertex> &vertex = graph->vertex;

        vector<std::vector<Mirror>> &mirror = graph->mirror;

        vector<double> score(graph->DC_num);

        int origin_dc = vertex[v].current_dc;
        int init_dc = vertex[v].init_dc;

        for (int dc = 0; dc < DC_num; dc++)
        {
            if (vertex[v].current_dc == dc)
                continue;

            double movecost = graph->movecost;
            vector<DataCenter> DC = graph->DC;

            if (origin_dc == init_dc)
            {
                movecost += MOVE_DATA_UNIT * DC[init_dc].UploadPrice;
            }
            else if (dc == init_dc)
            {
                movecost -= MOVE_DATA_UNIT * DC[init_dc].UploadPrice;
            }
            if (vertex[v].is_high_degree)
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_in_degree > 0)
                        DC[origin_dc].gather_download_wan -= DATA_UNIT;
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                }
                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;

                if (mirrorin > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT;

                if (mirror[dc][v].local_in_degree > 0)
                    DC[dc].gather_upload_wan -= DATA_UNIT;
                if (mirror[dc][v].local_out_degree > 0)
                    DC[dc].apply_download_wan -= DATA_UNIT;

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != dc && i != origin_dc && mirror[i][v].local_in_degree > 0)
                        DC[dc].gather_download_wan += DATA_UNIT;
                    if (i != dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[dc].apply_upload_wan += DATA_UNIT;
                }

                if (mirrorin > 0)
                    DC[dc].gather_download_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[dc].apply_upload_wan += DATA_UNIT;

                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[dc].apply_upload_wan -= DATA_UNIT;
                        }

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != dc)

                            {
                                if (mirror[dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                }

                if (vertex[v].local_in_degree > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT,
                        DC[dc].gather_download_wan += DATA_UNIT;
                if (vertex[v].local_out_degree > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT,
                        DC[dc].apply_upload_wan += DATA_UNIT;

                if (mirror[dc][v].in_use)
                {
                    // if (mirror[dc][v].local_in_degree > 0)
                    // DC[dc].gather_upload_wan -= DATA_UNIT;
                    if (mirror[dc][v].local_out_degree > 0)
                        DC[dc].apply_download_wan -= DATA_UNIT;
                }

                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;
                // mirror[dc][v].in_use = false;
                // mirror[dc][v].del();

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[dc].apply_upload_wan += DATA_UNIT;
                }

                for (auto &in_neighbour : vertex[v].in_edge)
                {
                    // mirror[origin_dc][v].local_in_degree--;
                    // vertex[v].local_in_degree++;
                    mirrorin--;
                    if (mirrorin == 0)
                    {
                        DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                        DC[dc].gather_download_wan -= DATA_UNIT;
                    }
                    // if (vertex[in_neighbour].is_high_degree)
                    {
                        int in_neighbour_dc = vertex[in_neighbour].current_dc;
                        if (in_neighbour_dc == origin_dc)
                        {
                            // vertex[in_neighbour].local_out_degree--;
                            if (mirror[dc][in_neighbour].local_out_degree == 0)
                            {
                                DC[dc].apply_download_wan += DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                            }
                            // mirror[dc][in_neighbour].add(0, 1);
                        }
                        else
                        {
                            // mirror[origin_dc][in_neighbour].local_out_degree--;
                            if (mirror[origin_dc][in_neighbour].local_out_degree == 1)
                            {
                                DC[origin_dc].apply_download_wan -= DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan -= DATA_UNIT;
                            }
                            if (in_neighbour_dc != dc)
                            {
                                if (mirror[dc][in_neighbour].local_out_degree == 0)
                                {
                                    DC[dc].apply_download_wan += DATA_UNIT;
                                    DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                                }
                                // mirror[dc][in_neighbour].add(0, 1);
                            }
                        }
                    }
                }
                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[dc].apply_upload_wan -= DATA_UNIT;
                        }

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != dc)

                            {
                                if (mirror[dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            double t, p;

            graph->calculate_network_time_price(t, p, DC);
            // printf("move %lld to %d : time > %.8f\tcost > %.8f\tmovecost > %.8f\n", v, dc, t, p, movecost);
            // if(t < graph->transfer_time)
            //     cout << "/*-" << endl;
            score[dc] = Score(graph->transfer_time, graph->transfer_cost, graph->movecost, t, p, movecost, it);
            // printf("%f %f %f %f\n", graph->transfer_time, graph->transfer_cost + graph->movecost, t, p + movecost);
            // cout << score[dc] << endl;
            // if(score[dc] > 0)
        }
        // DC[origin_dc].vertex_num--;
        // DC[dc].vertex_num++;
        // vertex[v].current_dc = dc;

        // calculate_network_time_price(transfer_time, transfer_cost);
        // if(v > 870000)
        // {
        // for( auto &x : score)
        //     cout << x << " ";
        // cout << endl;
        /*double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
        for (int i = 0; i < DC_num; i++)
        {
            graph->moveVertex(v, i);
            // cout << Score(old_time, old_cost, old_mvcost, graph->transfer_time, graph->transfer_cost, graph->movecost, it) << " ";
            if (fabs(Score(old_time, old_cost, old_mvcost, graph->transfer_time, graph->transfer_cost, graph->movecost, it) - score[i]) > 0.0000001)
                cout << "error : " << score[i] << ' ' << Score(old_time, old_cost, old_mvcost, graph->transfer_time, graph->transfer_cost, graph->movecost, it) << endl;
        }
        graph->moveVertex(v, origin_dc);*/
        // cout << endl<< endl;
        // }
        return score;
    }
    void update_prob(int it) // 更新概率
    {
#pragma omp parallel for
        for (id_type v = 0; v < graph->vertex_num; v++)
        {
            vector<double> s = moveVirtualVertex(v, it);
            int max_s = max_element(s.begin(), s.end()) - s.begin();
            for (int i = 0; i < graph->DC_num; i++)
            {
                if (max_s == i)
                    probability[v][i] += alpha * (1 - probability[v][i]);
                else
                    probability[v][i] = (1 - alpha) * probability[v][i];
            }
        }
    }
    void update_prob_and_make_roulette_choice(int it) // 更新概率
    {
#pragma omp parallel for
        for (id_type v = 0; v < graph->vertex_num; v++)
        {
            vector<double> s = moveVirtualVertex(v, it);
            int max_s = max_element(s.begin(), s.end()) - s.begin();
            for (int i = 0; i < graph->DC_num; i++)
            {
                if (max_s == i)
                    probability[v][i] += alpha * (1 - probability[v][i]);
                else
                    probability[v][i] = (1 - alpha) * probability[v][i];
            }
            pre_choice[v] = make_decision_roulette(v);
        }
    }
    void update_prob_and_make_greedy_choice(int it) // 更新概率
    {
#pragma omp parallel for
        for (id_type v = 0; v < graph->vertex_num; v++)
        {
            vector<double> s = moveVirtualVertex(v, it);
            int max_s = max_element(s.begin(), s.end()) - s.begin();
            for (int i = 0; i < graph->DC_num; i++)
            {
                if (max_s == i)
                    probability[v][i] += alpha * (1 - probability[v][i]);
                else
                    probability[v][i] = (1 - alpha) * probability[v][i];
            }
            pre_choice[v] = make_decision_greedy(v);
        }
    }
    void update_prob_and_make_greedy_choice(vector<id_type> &id, int it) // 更新概率
    {
#pragma omp parallel for
        for (int index = 0; index < id.size(); index++)
        {
            id_type v = id[index];
            vector<double> s = moveVirtualVertex(v, it);
            int max_s = max_element(s.begin(), s.end()) - s.begin();
            for (int i = 0; i < graph->DC_num; i++)
            {
                if (max_s == i)
                    probability[v][i] += alpha * (1 - probability[v][i]);
                else
                    probability[v][i] = (1 - alpha) * probability[v][i];
            }
            pre_choice[v] = make_decision_greedy(v);
        }
    }
    void sampling_update_prob_and_make_greedy_choice(vector<id_type> &id, int it, vector<bool> &id_trained)
    {
#pragma omp parallel for
        for (int index = 0; index < id.size(); index++)
        {
            double rand_pro = 1. * rand() / RAND_MAX;
            if (rand_pro <= sampling_rate)
            {
                id_trained[index] = true;
            }
            else
            {
                id_trained[index] = false;
                continue;
            }
            id_type v = id[index];
            vector<double> s = moveVirtualVertex(v, it);
            int max_s = max_element(s.begin(), s.end()) - s.begin();
            for (int i = 0; i < graph->DC_num; i++)
            {
                if (max_s == i)
                    probability[v][i] += alpha * (1 - probability[v][i]);
                else
                    probability[v][i] = (1 - alpha) * probability[v][i];
            }
            pre_choice[v] = make_decision_greedy(v);
        }
    }
    void train() // 开始训练
    {
        // auto &train_begin, train_end;
        auto train_begin = chrono::steady_clock::now();
        vector<id_type> id_vector(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        vector<id_type> train_id(batch_size);
        vector<int> origin_dc(batch_size);

        for (int it = 0; it < iteration; it++)
        {
            // clock_t iteration_begin, iteration_end;
            auto iteration_begin = chrono::steady_clock::now();
            ;
            // update_prob(it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = make_decision_roulette(v);
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    // graph->moveVertex(v, origin_dc);
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc != i)
                            probability[v][i] += alpha * (1 - probability[v][choice_dc]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
    }

    struct Mirror_Change
    {
    public:
        int dc;
        id_type v;
        int inD = 0;
        int outD = 0;
        int master_dc = -1;
    };
    void move_mirror(vector<Mirror_Change> &mirror_change)
    {
        //  cout << 123 << endl;
        // #pragma omp parallel for
        for (int a = 0; a < mirror_change.size(); a++)
        {
            // cout << omp_get_num_threads() << endl;
            int dc = mirror_change[a].dc;
            int v = mirror_change[a].v;
            int inD = mirror_change[a].inD;
            int outD = mirror_change[a].outD;
            int master_dc = mirror_change[a].master_dc == -1 ? graph->vertex[v].current_dc : mirror_change[a].master_dc;
            pthread_mutex_lock(&mirror_mutex[dc][v]);
            if (dc != master_dc)
            {

                if (graph->mirror[dc][v].local_in_degree <= 0 && inD > 0 && graph->mirror[dc][v].local_in_degree + inD > 0)
                {
                    //本来不需要上传的，现在需要了
                    atomic_gather_upload_wan[dc]++;          // mirror端发送
                    atomic_gather_download_wan[master_dc]++; // master端接收
                }

                else if (graph->mirror[dc][v].local_in_degree > 0 && inD < 0 && graph->mirror[dc][v].local_in_degree + inD <= 0)
                {
                    //本来不需要上传的，现在需要了
                    atomic_gather_upload_wan[dc]--;          // mirror端发送
                    atomic_gather_download_wan[master_dc]--; // master端接收
                }

                if (graph->mirror[dc][v].local_out_degree <= 0 && outD > 0 && graph->mirror[dc][v].local_out_degree + outD > 0)
                {
                    //本来不需要更新的，现在需要了
                    atomic_apply_download_wan[dc]++;      // mirror端接收
                    atomic_apply_upload_wan[master_dc]++; // master端发送
                }

                else if (graph->mirror[dc][v].local_out_degree > 0 && outD < 0 && graph->mirror[dc][v].local_out_degree + outD <= 0)
                {
                    //本来不需要更新的，现在需要了
                    atomic_apply_download_wan[dc]--;      // mirror端接收
                    atomic_apply_upload_wan[master_dc]--; // master端发送
                }

                graph->mirror[dc][v].local_in_degree += inD;
                graph->mirror[dc][v].local_out_degree += outD;

                if (graph->mirror[dc][v].local_in_degree <= 0 && graph->mirror[dc][v].local_out_degree <= 0)
                    graph->mirror[dc][v].in_use = false;
                else
                    graph->mirror[dc][v].in_use = true;
            }
            else
            {
                graph->vertex[v].local_in_degree += inD;
                graph->vertex[v].local_out_degree += outD;
            }
            pthread_mutex_unlock(&mirror_mutex[dc][v]);
        }
    }
    void train_vertex_parallel(vector<id_type> &train_id, int iter)
    {

        reset_virtual_DC();

        sem_t run_thread[batch_size];
        sem_t thread_ready[batch_size];

        // sem_init(&thread_ready[0], 0, 0);
        // sem_init(&run_thread[0], 0, 0);
        // sem_init(&thread_ready[1], 0, 0);
        // sem_init(&run_thread[1], 0, 0);
        for (int i = 0; i < batch_size; i++)
            sem_init(&thread_ready[i], 0, 0), sem_init(&run_thread[i], 0, 0);
        // int ptr = 0;
        vector<Mirror_Change> mirror_change;

        vector<Mirror_Change> master_change;
        // cout << train_id.size() << endl;
        double movecost;

#pragma omp parallel for num_threads(batch_size) schedule(static)
        for (int id_index = 0; id_index < train_id.size(); id_index++)
        {

            movecost = graph->movecost;
            vector<Vertex> &vertex = graph->vertex;
            vector<std::vector<Mirror>> &mirror = graph->mirror;
            // vector<DataCenter> DC = graph->DC;
            int &DC_num = graph->DC_num;

            double old_time = graph->transfer_time, old_cost = graph->transfer_cost;
            double mvcost_old = graph->movecost;

            // reset_virtual_DC();
            // int origin_dc[batch_size];
            vector<DataCenter> DC = graph->DC;

            id_type v = train_id[id_index];
            // cout << v << endl;
            int choice_dc = make_decision_greedy(v);
            int thread_num = omp_get_thread_num();
            // origin_dc[id_index] = vertex[v].current_dc;
            training_v[thread_num] = v;
            action[thread_num] = choice_dc;

            int origin_dc = vertex[v].current_dc;
            int init_dc = vertex[v].init_dc;

            if (origin_dc == init_dc)
            {
#pragma omp atomic
                movecost += MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }
            else if (choice_dc == init_dc)
            {
#pragma omp atomic
                movecost -= MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }
            if (vertex[v].is_high_degree)
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_in_degree > 0)
                        atomic_gather_download_wan[origin_dc] -= 1;
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        atomic_apply_upload_wan[origin_dc] -= 1;
                }
                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;

                if (mirrorin > 0)
                    atomic_gather_upload_wan[origin_dc] += 1;
                if (mirrorout > 0)
                    atomic_apply_download_wan[origin_dc] += 1;

                if (mirror[choice_dc][v].local_in_degree > 0)
                    atomic_gather_upload_wan[choice_dc] -= 1;
                if (mirror[choice_dc][v].local_out_degree > 0)
                    atomic_apply_download_wan[choice_dc] -= 1;

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_in_degree > 0)
                        atomic_gather_download_wan[choice_dc] += 1;
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        atomic_apply_upload_wan[choice_dc] += 1;
                }

                if (mirrorin > 0)
                    atomic_gather_download_wan[choice_dc] += 1;
                if (mirrorout > 0)
                    atomic_apply_upload_wan[choice_dc] += 1;

                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            atomic_apply_download_wan[origin_dc] -= 1;
                            atomic_apply_upload_wan[choice_dc] -= 1;
                        }

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                atomic_gather_upload_wan[choice_dc] += 1;
                                atomic_gather_download_wan[origin_dc] += 1;
                            }
                            // mirror[dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                atomic_gather_upload_wan[origin_dc] -= 1;
                                atomic_gather_download_wan[out_neighbour_dc] -= 1;
                            }
                            if (out_neighbour_dc != choice_dc)

                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    atomic_gather_upload_wan[choice_dc] += 1;
                                    atomic_gather_download_wan[out_neighbour_dc] += 1;
                                }
                                // mirror[dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        atomic_apply_upload_wan[origin_dc] -= 1;
                    if (mirror[i][v].in_use)
                    {
#pragma omp critical
                        {
                            master_change.push_back({i, v, -mirror[i][v].local_in_degree, -mirror[i][v].local_out_degree, origin_dc});
                            if (i != choice_dc)
                                master_change.push_back({i, v, mirror[i][v].local_in_degree, mirror[i][v].local_out_degree, choice_dc});
                        }
                    }
                }
#pragma omp critical
                {
                    master_change.push_back({choice_dc, v, mirror[choice_dc][v].local_in_degree - vertex[v].local_in_degree, mirror[choice_dc][v].local_out_degree - vertex[v].local_out_degree, choice_dc});
                    master_change.push_back({origin_dc, v, vertex[v].local_in_degree, vertex[v].local_out_degree, choice_dc});
                }
                if (vertex[v].local_in_degree > 0)
                    atomic_gather_upload_wan[origin_dc] += 1,
                        atomic_gather_download_wan[choice_dc] += 1;
                if (vertex[v].local_out_degree > 0)
                    atomic_apply_download_wan[origin_dc] += 1,
                        atomic_apply_upload_wan[choice_dc] += 1;

                if (mirror[choice_dc][v].in_use)
                {
                    // if (mirror[choice_dc][v].local_in_degree > 0)
                    // atomic_gather_upload_wan[choice_dc] -= 1;
                    if (mirror[choice_dc][v].local_out_degree > 0)
                        atomic_apply_download_wan[choice_dc] -= 1;
                }

                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;
                // mirror[dc][v].in_use = false;
                // mirror[dc][v].del();

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        atomic_apply_upload_wan[choice_dc] += 1;
                }

                for (auto &in_neighbour : vertex[v].in_edge)
                {
                    // mirror[origin_dc][v].local_in_degree--;
                    // vertex[v].local_in_degree++;
                    mirrorin--;
                    if (mirrorin == 0)
                    {
                        atomic_gather_upload_wan[origin_dc] -= 1;
                        atomic_gather_download_wan[choice_dc] -= 1;
                    }
#pragma omp critical
                    {
                        mirror_change.push_back({origin_dc, v, -1, 0, choice_dc});
                        mirror_change.push_back({choice_dc, v, 1, 0, choice_dc});
                        mirror_change.push_back({origin_dc, in_neighbour, 0, -1, -1});
                        mirror_change.push_back({choice_dc, in_neighbour, 0, 1, -1});
                    }
                    // if (vertex[in_neighbour].is_high_degree)
                    // cout << v << endl;
                    // cout << in_neighbour << endl;
                    // cout << vertex[in_neighbour].current_dc << endl;
                    int in_neighbour_dc = vertex[in_neighbour].current_dc;
                    if (in_neighbour_dc == origin_dc)
                    {
                        // vertex[in_neighbour].local_out_degree--;
                        if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                        {
                            atomic_apply_download_wan[choice_dc] += 1;
                            atomic_apply_upload_wan[in_neighbour_dc] += 1;
                        }
                        // mirror[dc][in_neighbour].add(0, 1);
                    }
                    else
                    {
                        // mirror[origin_dc][in_neighbour].local_out_degree--;
                        if (mirror[origin_dc][in_neighbour].local_out_degree == 1)
                        {
                            atomic_apply_download_wan[origin_dc] -= 1;
                            atomic_apply_upload_wan[in_neighbour_dc] -= 1;
                        }
                        if (in_neighbour_dc != choice_dc)
                        {
                            if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                            {
                                atomic_apply_download_wan[choice_dc] += 1;
                                atomic_apply_upload_wan[in_neighbour_dc] += 1;
                            }
                            // mirror[dc][in_neighbour].add(0, 1);
                        }
                    }
                }
                for (auto &out_neighbour : vertex[v].out_edge)
                {

                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            atomic_apply_download_wan[origin_dc] -= 1;
                            atomic_apply_upload_wan[choice_dc] -= 1;
                        }
#pragma omp critical
                        {
                            mirror_change.push_back({origin_dc, v, 0, -1, choice_dc});
                            mirror_change.push_back({choice_dc, v, 0, 1, choice_dc});
                            mirror_change.push_back({origin_dc, out_neighbour, -1, 0, -1});
                            mirror_change.push_back({choice_dc, out_neighbour, 1, 0, -1});
                        }

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                atomic_gather_upload_wan[choice_dc] += 1;
                                atomic_gather_download_wan[origin_dc] += 1;
                            }
                            // mirror[dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                atomic_gather_upload_wan[origin_dc] -= 1;
                                atomic_gather_download_wan[out_neighbour_dc] -= 1;
                            }
                            if (out_neighbour_dc != choice_dc)
                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    atomic_gather_upload_wan[choice_dc] += 1;
                                    atomic_gather_download_wan[out_neighbour_dc] += 1;
                                }
                                // mirror[dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            sem_post(&thread_ready[thread_num]);
            if (omp_get_thread_num() == 0)
            {
                // cout << omp_get_num_threads() << endl;
                for (int i = 0; i < batch_size; i++)
                    sem_wait(&thread_ready[i]);
                // cout << 345 << endl;
                for (int i = 0; i < graph->DC_num; i++)
                {
                    DC[i].apply_download_wan += atomic_apply_download_wan[i] * DATA_UNIT;
                    DC[i].apply_upload_wan += atomic_apply_upload_wan[i] * DATA_UNIT;
                    DC[i].gather_download_wan += atomic_gather_download_wan[i] * DATA_UNIT;
                    DC[i].gather_upload_wan += atomic_gather_upload_wan[i] * DATA_UNIT;
                }

                double t, p;

                graph->calculate_network_time_price(t, p, DC);

                reset_virtual_DC();

                if (Signal(graph->transfer_time, graph->transfer_cost, graph->movecost, t, p, movecost, iter))
                {
                    // cout << 123 << endl;
                    // for (int i = 0; i < batch_size; i++)
                    // graph->moveVertex(training_v[i], action[i]);
                    for (int i = 0; i < batch_size; i++)
                    {
                        graph->DC[graph->vertex[training_v[i]].current_dc].vertex_num--;
                        graph->vertex[training_v[i]].current_dc = action[i];
                        graph->DC[action[i]].vertex_num++;
                    }

                    graph->movecost = movecost;

                    move_mirror(master_change);
                    move_mirror(mirror_change);
                    // cout << 123 << endl;

                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        graph->DC[i].apply_download_wan += atomic_apply_download_wan[i] * DATA_UNIT;
                        graph->DC[i].apply_upload_wan += atomic_apply_upload_wan[i] * DATA_UNIT;
                        graph->DC[i].gather_download_wan += atomic_gather_download_wan[i] * DATA_UNIT;
                        graph->DC[i].gather_upload_wan += atomic_gather_upload_wan[i] * DATA_UNIT;
                    }
                    //

                    graph->calculate_network_time_price(graph->transfer_time, graph->transfer_cost);
                    reset_virtual_DC();
                }

                mirror_change.clear();
                master_change.clear();

                // sem_init(&run_thread, 0, batch_size);
                for (int i = 0; i < batch_size; i++)
                    sem_post(&run_thread[i]);
                // cout << 222 << endl;
                // cout << id_index << endl;
            }
            sem_wait(&run_thread[thread_num]);
        }
    }

    void train_vertex_parallel2(vector<id_type> &train_id, int iter)
    {

        reset_virtual_DC();

        sem_t run_thread[batch_size];
        sem_t thread_ready[batch_size];

        // sem_init(&thread_ready[0], 0, 0);
        // sem_init(&run_thread[0], 0, 0);
        // sem_init(&thread_ready[1], 0, 0);
        // sem_init(&run_thread[1], 0, 0);
        for (int i = 0; i < batch_size; i++)
            sem_init(&thread_ready[i], 0, 0), sem_init(&run_thread[i], 0, 0);
        // int ptr = 0;

        // cout << train_id.size() << endl;
        vector<double> mv_cost(batch_size, 0);

#pragma omp parallel for num_threads(batch_size) schedule(static)
        for (int id_index = 0; id_index < train_id.size(); id_index++)
        {

            double movecost = graph->movecost;
            vector<Vertex> &vertex = graph->vertex;
            vector<std::vector<Mirror>> &mirror = graph->mirror;
            // vector<DataCenter> DC = graph->DC;
            int &DC_num = graph->DC_num;

            double old_time = graph->transfer_time, old_cost = graph->transfer_cost;
            double mvcost_old = graph->movecost;

            // reset_virtual_DC();
            // int origin_dc[batch_size];
            vector<DataCenter> DC = graph->DC;
            vector<Mirror_Change> mirror_change;

            vector<Mirror_Change> master_change;

            id_type v = train_id[id_index];
            // cout << v << endl;
            int choice_dc = make_decision_greedy(v);
            int thread_num = omp_get_thread_num();
            // origin_dc[id_index] = vertex[v].current_dc;
            training_v[thread_num] = v;
            action[thread_num] = choice_dc;

            int origin_dc = vertex[v].current_dc;
            int init_dc = vertex[v].init_dc;

            if (origin_dc == init_dc)
            {
                movecost += MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }
            else if (choice_dc == init_dc)
            {
                movecost -= MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }

            if (vertex[v].is_high_degree)
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use)
                    {
                        master_change.push_back({i, v, -mirror[i][v].local_in_degree, -mirror[i][v].local_out_degree, origin_dc});
                        if (i != choice_dc)
                            master_change.push_back({i, v, mirror[i][v].local_in_degree, mirror[i][v].local_out_degree, choice_dc});

                        if (mirror[i][v].local_in_degree > 0)
                            DC[origin_dc].gather_download_wan -= DATA_UNIT;

                        if (mirror[i][v].local_out_degree > 0)
                            DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                    }
                }
                master_change.push_back({choice_dc, v, mirror[choice_dc][v].local_in_degree - vertex[v].local_in_degree, mirror[choice_dc][v].local_out_degree - vertex[v].local_out_degree, choice_dc});
                master_change.push_back({origin_dc, v, vertex[v].local_in_degree, vertex[v].local_out_degree, choice_dc});

                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;

                if (mirrorin > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT;

                if (mirror[choice_dc][v].local_in_degree > 0)
                    DC[choice_dc].gather_upload_wan -= DATA_UNIT;
                if (mirror[choice_dc][v].local_out_degree > 0)
                    DC[choice_dc].apply_download_wan -= DATA_UNIT;

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_in_degree > 0)
                        DC[choice_dc].gather_download_wan += DATA_UNIT;
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;
                }

                if (mirrorin > 0)
                    DC[choice_dc].gather_download_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[choice_dc].apply_upload_wan += DATA_UNIT;

                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[choice_dc].apply_upload_wan -= DATA_UNIT;
                        }
                        mirror_change.push_back({origin_dc, v, 0, -1, choice_dc});
                        mirror_change.push_back({choice_dc, v, 0, 1, choice_dc});
                        mirror_change.push_back({origin_dc, out_neighbour, -1, 0, -1});
                        mirror_change.push_back({choice_dc, out_neighbour, 1, 0, -1});

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != choice_dc)

                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                    if (mirror[i][v].in_use)
                    {
                        master_change.push_back({i, v, -mirror[i][v].local_in_degree, -mirror[i][v].local_out_degree, origin_dc});
                        if (i != choice_dc)
                            master_change.push_back({i, v, mirror[i][v].local_in_degree, mirror[i][v].local_out_degree, choice_dc});
                    }
                }

                master_change.push_back({choice_dc, v, mirror[choice_dc][v].local_in_degree - vertex[v].local_in_degree, mirror[choice_dc][v].local_out_degree - vertex[v].local_out_degree, choice_dc});
                master_change.push_back({origin_dc, v, vertex[v].local_in_degree, vertex[v].local_out_degree, choice_dc});

                if (vertex[v].local_in_degree > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT,
                        DC[choice_dc].gather_download_wan += DATA_UNIT;
                if (vertex[v].local_out_degree > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT,
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;

                if (mirror[choice_dc][v].in_use)
                {
                    // if (mirror[choice_dc][v].local_in_degree > 0)
                    // DC[choice_dc].gather_upload_wan -= DATA_UNIT;
                    if (mirror[choice_dc][v].local_out_degree > 0)
                        DC[choice_dc].apply_download_wan -= DATA_UNIT;
                }

                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;
                // mirror[choice_dc][v].in_use = false;
                // mirror[choice_dc][v].del();

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;
                }

                for (auto &in_neighbour : vertex[v].in_edge)
                {
                    // mirror[origin_dc][v].local_in_degree--;
                    // vertex[v].local_in_degree++;
                    mirrorin--;
                    if (mirrorin == 0)
                    {
                        DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                        DC[choice_dc].gather_download_wan -= DATA_UNIT;
                    }
                    mirror_change.push_back({origin_dc, v, -1, 0, choice_dc});
                    mirror_change.push_back({choice_dc, v, 1, 0, choice_dc});
                    mirror_change.push_back({origin_dc, in_neighbour, 0, -1, -1});
                    mirror_change.push_back({choice_dc, in_neighbour, 0, 1, -1});

                    // if (vertex[in_neighbour].is_high_degree)
                    {
                        int in_neighbour_dc = vertex[in_neighbour].current_dc;
                        if (in_neighbour_dc == origin_dc)
                        {
                            // vertex[in_neighbour].local_out_degree--;
                            if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                            {
                                DC[choice_dc].apply_download_wan += DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][in_neighbour].add(0, 1);
                        }
                        else
                        {
                            // mirror[origin_dc][in_neighbour].local_out_degree--;
                            if (mirror[origin_dc][in_neighbour].local_out_degree == 1)
                            {
                                DC[origin_dc].apply_download_wan -= DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan -= DATA_UNIT;
                            }
                            if (in_neighbour_dc != choice_dc)
                            {
                                if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                                {
                                    DC[choice_dc].apply_download_wan += DATA_UNIT;
                                    DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][in_neighbour].add(0, 1);
                            }
                        }
                    }
                }
                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[choice_dc].apply_upload_wan -= DATA_UNIT;
                        }
                        mirror_change.push_back({origin_dc, v, 0, -1, choice_dc});
                        mirror_change.push_back({choice_dc, v, 0, 1, choice_dc});
                        mirror_change.push_back({origin_dc, out_neighbour, -1, 0, -1});
                        mirror_change.push_back({choice_dc, out_neighbour, 1, 0, -1});

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != choice_dc)

                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }

            double t, p;
            bool do_action = false;

            graph->calculate_network_time_price(t, p, DC);
            if (Signal(graph->transfer_time, graph->transfer_cost, graph->movecost, t, p, movecost, iter))
            {
                mv_cost[thread_num] = movecost - graph->movecost;
                do_action = true;

                graph->DC[graph->vertex[v].current_dc].vertex_num--;
                graph->vertex[v].current_dc = choice_dc;
                graph->DC[choice_dc].vertex_num++;
            }
            else
                mv_cost[thread_num] = 0;

            sem_post(&thread_ready[thread_num]);
            if (omp_get_thread_num() == 0)
            {
                for (int i = 0; i < batch_size; i++)
                    sem_wait(&thread_ready[i]);
                for (int i = 0; i < batch_size; i++)
                    sem_post(&run_thread[i]);
            }
            sem_wait(&run_thread[thread_num]);
            if (do_action)
            {
                move_mirror(master_change);
                move_mirror(mirror_change);
            }
            sem_post(&thread_ready[thread_num]);

            if (omp_get_thread_num() == 0)
            {
                for (int i = 0; i < batch_size; i++)
                    sem_wait(&thread_ready[i]);

                for (int i = 0; i < graph->DC_num; i++)
                {
                    graph->DC[i].apply_download_wan += atomic_apply_download_wan[i] * DATA_UNIT;
                    graph->DC[i].apply_upload_wan += atomic_apply_upload_wan[i] * DATA_UNIT;
                    graph->DC[i].gather_download_wan += atomic_gather_download_wan[i] * DATA_UNIT;
                    graph->DC[i].gather_upload_wan += atomic_gather_upload_wan[i] * DATA_UNIT;
                }
                //
                for (int i = 0; i < batch_size; i++)
                    graph->movecost += mv_cost[i];

                graph->calculate_network_time_price(graph->transfer_time, graph->transfer_cost);
                reset_virtual_DC();
                for (int i = 0; i < batch_size; i++)
                    sem_post(&run_thread[i]);
            }
            sem_wait(&run_thread[thread_num]);
        }
    }

    void train_vertex_parallel3(vector<id_type> &train_id, int iter)
    {

        reset_virtual_DC();

        sem_t run_thread[batch_size];
        sem_t thread_ready[batch_size];

        // sem_init(&thread_ready[0], 0, 0);
        // sem_init(&run_thread[0], 0, 0);
        // sem_init(&thread_ready[1], 0, 0);
        // sem_init(&run_thread[1], 0, 0);
        for (int i = 0; i < batch_size; i++)
            sem_init(&thread_ready[i], 0, 0), sem_init(&run_thread[i], 0, 0);
        // int ptr = 0;

        // cout << train_id.size() << endl;
        vector<double> mv_cost(batch_size, 0);

#pragma omp parallel for num_threads(batch_size) schedule(static)
        for (int id_index = 0; id_index < train_id.size(); id_index++)
        {

            double movecost = graph->movecost;
            vector<Vertex> &vertex = graph->vertex;
            vector<std::vector<Mirror>> &mirror = graph->mirror;
            // vector<DataCenter> DC = graph->DC;
            int &DC_num = graph->DC_num;

            double old_time = graph->transfer_time, old_cost = graph->transfer_cost;
            double mvcost_old = graph->movecost;

            // reset_virtual_DC();
            // int origin_dc[batch_size];
            vector<DataCenter> DC = graph->DC;

            id_type v = train_id[id_index];
            // cout << v << endl;
            int choice_dc = make_decision_greedy(v);
            int thread_num = omp_get_thread_num();
            // origin_dc[id_index] = vertex[v].current_dc;
            training_v[thread_num] = v;
            action[thread_num] = choice_dc;

            int origin_dc = vertex[v].current_dc;
            int init_dc = vertex[v].init_dc;

            if (origin_dc == init_dc)
            {
                movecost += MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }
            else if (choice_dc == init_dc)
            {
                movecost -= MOVE_DATA_UNIT * graph->DC[init_dc].UploadPrice;
            }

            if (vertex[v].is_high_degree)
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use)
                    {

                        if (mirror[i][v].local_in_degree > 0)
                            DC[origin_dc].gather_download_wan -= DATA_UNIT;

                        if (mirror[i][v].local_out_degree > 0)
                            DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                    }
                }
                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;

                if (mirrorin > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT;

                if (mirror[choice_dc][v].local_in_degree > 0)
                    DC[choice_dc].gather_upload_wan -= DATA_UNIT;
                if (mirror[choice_dc][v].local_out_degree > 0)
                    DC[choice_dc].apply_download_wan -= DATA_UNIT;

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_in_degree > 0)
                        DC[choice_dc].gather_download_wan += DATA_UNIT;
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;
                }

                if (mirrorin > 0)
                    DC[choice_dc].gather_download_wan += DATA_UNIT;
                if (mirrorout > 0)
                    DC[choice_dc].apply_upload_wan += DATA_UNIT;

                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[choice_dc].apply_upload_wan -= DATA_UNIT;
                        }
                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != choice_dc)

                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < DC_num; i++)
                {
                    if (mirror[i][v].in_use && mirror[i][v].local_out_degree > 0)
                        DC[origin_dc].apply_upload_wan -= DATA_UNIT;
                }

                if (vertex[v].local_in_degree > 0)
                    DC[origin_dc].gather_upload_wan += DATA_UNIT,
                        DC[choice_dc].gather_download_wan += DATA_UNIT;
                if (vertex[v].local_out_degree > 0)
                    DC[origin_dc].apply_download_wan += DATA_UNIT,
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;

                if (mirror[choice_dc][v].in_use)
                {
                    // if (mirror[choice_dc][v].local_in_degree > 0)
                    // DC[choice_dc].gather_upload_wan -= DATA_UNIT;
                    if (mirror[choice_dc][v].local_out_degree > 0)
                        DC[choice_dc].apply_download_wan -= DATA_UNIT;
                }

                int mirrorin = vertex[v].local_in_degree;
                int mirrorout = vertex[v].local_out_degree;
                // mirror[choice_dc][v].in_use = false;
                // mirror[choice_dc][v].del();

                for (int i = 0; i < DC_num; i++)
                {
                    if (i != choice_dc && i != origin_dc && mirror[i][v].local_out_degree > 0)
                        DC[choice_dc].apply_upload_wan += DATA_UNIT;
                }

                for (auto &in_neighbour : vertex[v].in_edge)
                {
                    // mirror[origin_dc][v].local_in_degree--;
                    // vertex[v].local_in_degree++;
                    mirrorin--;
                    if (mirrorin == 0)
                    {
                        DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                        DC[choice_dc].gather_download_wan -= DATA_UNIT;
                    }

                    // if (vertex[in_neighbour].is_high_degree)
                    {
                        int in_neighbour_dc = vertex[in_neighbour].current_dc;
                        if (in_neighbour_dc == origin_dc)
                        {
                            // vertex[in_neighbour].local_out_degree--;
                            if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                            {
                                DC[choice_dc].apply_download_wan += DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][in_neighbour].add(0, 1);
                        }
                        else
                        {
                            // mirror[origin_dc][in_neighbour].local_out_degree--;
                            if (mirror[origin_dc][in_neighbour].local_out_degree == 1)
                            {
                                DC[origin_dc].apply_download_wan -= DATA_UNIT;
                                DC[in_neighbour_dc].apply_upload_wan -= DATA_UNIT;
                            }
                            if (in_neighbour_dc != choice_dc)
                            {
                                if (mirror[choice_dc][in_neighbour].local_out_degree == 0)
                                {
                                    DC[choice_dc].apply_download_wan += DATA_UNIT;
                                    DC[in_neighbour_dc].apply_upload_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][in_neighbour].add(0, 1);
                            }
                        }
                    }
                }
                for (auto &out_neighbour : vertex[v].out_edge)
                {
                    if (vertex[out_neighbour].is_high_degree)
                    {
                        mirrorout--;
                        if (mirrorout == 0)
                        {
                            DC[origin_dc].apply_download_wan -= DATA_UNIT;
                            DC[choice_dc].apply_upload_wan -= DATA_UNIT;
                        }

                        if (vertex[out_neighbour].current_dc == origin_dc)
                        {
                            // vertex[out_neighbour].local_in_degree--;

                            if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                            {
                                DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                DC[origin_dc].gather_download_wan += DATA_UNIT;
                            }
                            // mirror[choice_dc][out_neighbour].add(1, 0);
                        }
                        else
                        {
                            int out_neighbour_dc = vertex[out_neighbour].current_dc;
                            // mirror[origin_dc][out_neighbour].local_in_degree--;
                            if (mirror[origin_dc][out_neighbour].local_in_degree == 1)
                            {
                                DC[origin_dc].gather_upload_wan -= DATA_UNIT;
                                DC[out_neighbour_dc].gather_download_wan -= DATA_UNIT;
                            }
                            if (out_neighbour_dc != choice_dc)

                            {
                                if (mirror[choice_dc][out_neighbour].local_in_degree == 0)
                                {
                                    DC[choice_dc].gather_upload_wan += DATA_UNIT;
                                    DC[out_neighbour_dc].gather_download_wan += DATA_UNIT;
                                }
                                // mirror[choice_dc][out_neighbour].add(1, 0);
                            }
                        }
                    }
                }
            }

            double t, p;

            graph->calculate_network_time_price(t, p, DC);

#pragma omp critical
            {
                if (Signal(graph->transfer_time, graph->transfer_cost, graph->movecost, t, p, movecost, iter))
                {
                    graph->movecost += movecost - graph->movecost;

                    graph->moveVertex(v, choice_dc);
                }
            }

            sem_post(&thread_ready[thread_num]);

            if (omp_get_thread_num() == 0)
            {
                for (int i = 0; i < batch_size; i++)
                    sem_wait(&thread_ready[i]);

                graph->calculate_network_time_price(graph->transfer_time, graph->transfer_cost);

                for (int i = 0; i < batch_size; i++)
                    sem_post(&run_thread[i]);
            }
            sem_wait(&run_thread[thread_num]);
        }
    }

    void train_vertex_parallel4(vector<id_type> &train_id, int iter)
    {

        reset_virtual_DC();

        sem_t run_thread[batch_size];
        sem_t thread_ready[batch_size];

        // sem_init(&thread_ready[0], 0, 0);
        // sem_init(&run_thread[0], 0, 0);
        // sem_init(&thread_ready[1], 0, 0);
        // sem_init(&run_thread[1], 0, 0);
        for (int i = 0; i < batch_size; i++)
            sem_init(&thread_ready[i], 0, 0), sem_init(&run_thread[i], 0, 0);
        // int ptr = 0;

        // cout << train_id.size() << endl;
        vector<double> mv_cost(batch_size, 0);

#pragma omp parallel for num_threads(batch_size) schedule(static)
        for (int id_index = 0; id_index < train_id.size(); id_index++)
        {
            int thread_num = omp_get_thread_num();
#pragma omp critical
            {
                int v = train_id[id_index];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = make_decision_roulette(v);
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, iter);
                if (!signal)
                    graph->moveVertex(v, origin_dc);
            }

            sem_post(&thread_ready[thread_num]);

            if (omp_get_thread_num() == 0)
            {
                for (int i = 0; i < batch_size; i++)
                    sem_wait(&thread_ready[i]);

                for (int i = 0; i < batch_size; i++)
                    sem_post(&run_thread[i]);
            }
            sem_wait(&run_thread[thread_num]);
        }
    }

    void train_parallel() // 开始训练
    {
        // auto &train_begin, train_end;
        auto train_begin = chrono::steady_clock::now();

        vector<id_type> id_vector(graph->vertex_num - graph->vertex_num % batch_size);
        for (int i = 0; i < graph->vertex_num - graph->vertex_num % batch_size; i++)
            id_vector[i] = i;

        vector<id_type> train_id(batch_size);
        vector<int> origin_dc(batch_size);

        for (int it = 0; it < iteration; it++)
        {
            // clock_t iteration_begin, iteration_end;
            auto iteration_begin = chrono::steady_clock::now();
            ;
            update_prob(it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            train_vertex_parallel4(id_vector, it);
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            // graph->print();
        }
        graph->print();
        auto train_end = chrono::steady_clock::now();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
    }

    void baseline_LA()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        auto train_begin = chrono::steady_clock::now();
        id_vector = vector<id_type>(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            update_prob_and_make_roulette_choice(it);
            // update_prob_and_make_greedy_choice(it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    // graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Penalty()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        auto train_begin = chrono::steady_clock::now();
        id_vector = vector<id_type>(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            update_prob_and_make_roulette_choice(it);
            // update_prob_and_make_greedy_choice(it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    // graph->moveVertex(v, origin_dc);
                    double belta = 0.2;
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc != i)
                            probability[v][i] = (1 - belta) * probability[v][i] + belta / (graph->DC_num - 1);
                        else
                            probability[v][i] = (1 - belta) * probability[v][choice_dc];
                    }
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        auto train_begin = chrono::steady_clock::now();
        id_vector = vector<id_type>(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            update_prob_and_make_roulette_choice(it);
            // update_prob_and_make_greedy_choice(it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;
        sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b){return graph->vertex[a].in_degree > graph->vertex[b].in_degree;});
        // printf("%d -> %d\n", graph->vertex[id_vector[0]].out_degree, graph->vertex[id_vector[id_vector.size()-1]].out_degree);
        auto train_begin = chrono::steady_clock::now();

        double iteration_use = 0;

        for (int it = 0; it < iteration; it++)
        {

            // 打乱顺序
            // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            // random_shuffle(id_vector.begin(), id_vector.end());

            auto iteration_begin = chrono::steady_clock::now();

            // update_prob_and_make_roulette_choice(it);
            update_prob_and_make_greedy_choice(it);

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << end
            iteration_use += (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        printf("[LeanrningAutomaton] training use : %f s\n", iteration_use);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_HighDegreeNode()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->high_degree_node_num);
        int p = 0;
        for (int i = 0; i < graph->vertex_num; i++)
        {
            if (graph->vertex[i].is_high_degree)
                id_vector[p++] = i;
        }

        auto train_begin = chrono::steady_clock::now();

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            // update_prob_and_make_roulette_choice(it);
            update_prob_and_make_greedy_choice(id_vector, it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_LowDegreeNode()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num - graph->high_degree_node_num);
        int p = 0;
        for (int i = 0; i < graph->vertex_num; i++)
        {
            if (!graph->vertex[i].is_high_degree)
                id_vector[p++] = i;
        }

        auto train_begin = chrono::steady_clock::now();

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            // update_prob_and_make_roulette_choice(it);
            update_prob_and_make_greedy_choice(id_vector, it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_MDegreeNode()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        // id_vector = vector<id_type> (graph->vertex_num - graph->high_degree_node_num);
        int p = 0;
        for (int i = 0; i < graph->vertex_num; i++)
        {
            if (graph->vertex[i].in_degree < graph->DC_num)
                id_vector.push_back(i);
        }

        auto train_begin = chrono::steady_clock::now();

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            // update_prob_and_make_roulette_choice(it);
            update_prob_and_make_greedy_choice(id_vector, it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_TOPKDegreeNode()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
        random_shuffle(id_vector.begin(), id_vector.end());

        sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b)
             { return graph->vertex[a].in_degree < graph->vertex[b].in_degree; });
        unsigned long long num = train_rate * graph->vertex_num;
        id_vector.resize(num);
        int max_in_degree = graph->vertex[id_vector[id_vector.size() - 1]].in_degree;

        auto train_begin = chrono::steady_clock::now();

        for (int it = 0; it < iteration; it++)
        {
            auto iteration_begin = chrono::steady_clock::now();
            // update_prob_and_make_roulette_choice(it);
            update_prob_and_make_greedy_choice(id_vector, it);

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            ;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;

        log_file << train_rate << '\t' << num << '\t' << max_in_degree << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_Sampling()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num);
        vector<bool> id_vector_trained(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;
        // sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b){return graph->vertex[a].out_degree > graph->vertex[b].out_degree;});
        // printf("%d -> %d\n", graph->vertex[id_vector[0]].out_degree, graph->vertex[id_vector[id_vector.size()-1]].out_degree);

        vector<double> sampling_rate_per_iteration(iteration);
        vector<double> overhead_per_iteration(iteration);
        vector<double> samplingrate_overhead_rate(iteration);
        double left_overhead = overhead_limit;
        double overhead_use = 0;

        // auto train_begin = chrono::steady_clock::now();

        for (int it = 0; it < iteration; it++)
        {
            // update_prob_and_make_roulette_choice(it);
            sampling_rate_per_iteration[it] = sampling_rate;

            // 打乱顺序
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            random_shuffle(id_vector.begin(), id_vector.end());

            auto iteration_begin = chrono::steady_clock::now();
            sampling_update_prob_and_make_greedy_choice(id_vector, it, id_vector_trained);

            for (int i = 0; i < id_vector.size(); i++)
            {
                if (!id_vector_trained[i])
                    continue;
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                    // for (int i = 0; i < graph->DC_num; i++)
                    // {
                    //     if (choice_dc != i)
                    //         probability[v][i] += alpha * (1 - probability[v][choice_dc]) / (graph->DC_num - 1);
                    //     else
                    //         probability[v][i] = (1 - alpha) * probability[v][choice_dc];
                    // }
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            double iteration_use_overhead = (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000;
            overhead_per_iteration[it] = iteration_use_overhead;
            overhead_use += iteration_use_overhead;
            samplingrate_overhead_rate[it] = sampling_rate / iteration_use_overhead;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] sampling rate : %f\n", sampling_rate);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", iteration_use_overhead);
            // cout << cnt << endl;
            left_overhead -= iteration_use_overhead;
            update_sampling_rate(it, samplingrate_overhead_rate, left_overhead);
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        // printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        printf("[LeanrningAutomaton] training use : %f s\n", overhead_use);
        // log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
        log_file << overhead_use << endl;
    }
    void update_sampling_rate(int cur_it, vector<double> &x, double left_overhead)
    {
        if (!auto_sampling_rate)
            return;
        int a = 1;
        int min_a = min(cur_it + 1, a);
        double sum = 0;
        for (int i = 0; i < min_a; i++)
            sum += x[cur_it - i];
        double left_time_per_iteration = left_overhead / (iteration - cur_it - 1);
        sampling_rate = max(0., sum / min_a * left_time_per_iteration);
    }
    void baseline_LA_with_Cancel_with_Greedy_Block()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num);
        vector<bool> id_vector_trained(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b)
             { return graph->vertex[a].in_degree < graph->vertex[b].in_degree; });
        // printf("%d -> %d\n", graph->vertex[id_vector[0]].in_degree, graph->vertex[id_vector[id_vector.size()-1]].in_degree);

        double overhead_use = 0;

        vector<vector<id_type>> BLOCKS;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
        // random_shuffle(id_vector.begin(), id_vector.end());

        // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
        // random_shuffle(id_vector.begin(), id_vector.end());

        // 按照Low High degree区分blocks

        /*
        BLOCKS.emplace_back(vector<id_type>());
        BLOCKS.emplace_back(vector<id_type>());
        for (id_type i = 0; i < graph->vertex_num; i++)
        {
            if (graph->vertex[i].is_high_degree)
                BLOCKS[1].push_back(i);
            else
                BLOCKS[0].push_back(i);
        }
        */

        int bnum = graph->vertex_num / block_nums;
        for (int i = 0; i < block_nums; i++)
        {
            vector<id_type> tmp;
            tmp.reserve(bnum + block_nums - 1);
            BLOCKS.emplace_back(tmp);
        }
        for (int i = 0; i < graph->vertex_num; i++)
            BLOCKS[min(i / bnum, block_nums - 1)].push_back(i);

        // auto train_begin = chrono::steady_clock::now();
        // double tmp_Budget = Budget / block_nums;
        // Budget = 0;
        for (auto &block_id : BLOCKS)
        {
            cout << "New block : " << block_id.size() << " vertexes.\n";
            // Budget+=tmp_Budget;
            for (int it = 0; it < iteration; it++)
            {
                // update_prob_and_make_roulette_choice(it);

                // 打乱顺序
                seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::shuffle(block_id.begin(), block_id.end(), std::default_random_engine(seed));
                random_shuffle(block_id.begin(), block_id.end());

                auto iteration_begin = chrono::steady_clock::now();
                update_prob_and_make_greedy_choice(block_id, it);

                for (int i = 0; i < block_id.size(); i++)
                {
                    id_type v = block_id[i];
                    int origin_dc = graph->vertex[v].current_dc;
                    double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                    int choice_dc = pre_choice[v];
                    graph->moveVertex(v, choice_dc);
                    double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                    bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                    if (signal)
                    {
                        for (int i = 0; i < graph->DC_num; i++)
                        {
                            if (choice_dc == i)
                                probability[v][i] += alpha * (1 - probability[v][i]);
                            else
                                probability[v][i] = (1 - alpha) * probability[v][i];
                        }
                    }
                    else
                    {
                        graph->moveVertex(v, origin_dc);
                    }
                }
                auto iteration_end = chrono::steady_clock::now();
                double iteration_use_overhead = (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000;
                overhead_use += iteration_use_overhead;
                printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
                printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
                printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
                printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
                printf("[LearningAutomaton] iteration use : %f s\n", iteration_use_overhead);
                // cout << cnt << endl;
                log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
            }
        }
        auto train_end = chrono::steady_clock::now();
        graph->print();
        // printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        printf("[LeanrningAutomaton] training use : %f s\n", overhead_use);
        // log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
        log_file << overhead_use << endl;
    }
    void baseline_LA_with_Cancel_with_Greedy_Mirror_Seq()
    {
        // auto &train_begin, train_end;
        log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << 0 << endl;
        id_vector = vector<id_type>(graph->vertex_num);
        vector<bool> id_vector_trained(graph->vertex_num);
        for (int i = 0; i < graph->vertex_num; i++)
            id_vector[i] = i;

        // sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b)
            //  { return graph->vertex[a].num_mirror < graph->vertex[b].num_mirror; });
        // printf("%d -> %d\n", graph->vertex[id_vector[0]].in_degree, graph->vertex[id_vector[id_vector.size()-1]].in_degree);

        double overhead_use = 0;

        


        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
        // random_shuffle(id_vector.begin(), id_vector.end());

        // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
        // random_shuffle(id_vector.begin(), id_vector.end());

        
        

        
        for (int it = 0; it < iteration; it++)
        {
            // update_prob_and_make_roulette_choice(it);
            
            // 打乱顺序
            // seed = std::chrono::system_clock::now().time_since_epoch().count();
            // std::shuffle(id_vector.begin(), id_vector.end(), std::default_random_engine(seed));
            // random_shuffle(id_vector.begin(), id_vector.end());
            // check_mirror();

            auto iteration_begin = chrono::steady_clock::now();

            sort(id_vector.begin(), id_vector.end(), [&](id_type a, id_type b)
             { return graph->vertex[a].num_mirror < graph->vertex[b].num_mirror; });
        
            
            update_prob_and_make_greedy_choice(id_vector, it);
            
            for (int i = 0; i < id_vector.size(); i++)
            {
                id_type v = id_vector[i];
                int origin_dc = graph->vertex[v].current_dc;
                double old_time = graph->transfer_time, old_cost = graph->transfer_cost, old_mvcost = graph->movecost;
                int choice_dc = pre_choice[v];
                graph->moveVertex(v, choice_dc);
                
                double new_time = graph->transfer_time, new_cost = graph->transfer_cost, new_mvcost = graph->movecost;
                bool signal = Signal(old_time, old_cost, old_mvcost, new_time, new_cost, new_mvcost, it);

                if (signal)
                {
                    for (int i = 0; i < graph->DC_num; i++)
                    {
                        if (choice_dc == i)
                            probability[v][i] += alpha * (1 - probability[v][i]);
                        else
                            probability[v][i] = (1 - alpha) * probability[v][i];
                    }
                }
                else
                {
                    graph->moveVertex(v, origin_dc);
                }
            }
            auto iteration_end = chrono::steady_clock::now();
            double iteration_use_overhead = (double)std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_begin).count() / 1000;
            overhead_use += iteration_use_overhead;
            printf("\n\n[LearningAutomaton] iteration : %d / %d\n", it + 1, iteration);
            printf("[LearningAutomaton] time : %f (%f)\n", graph->transfer_time, graph->transfer_time / origin_time);
            printf("[LearningAutomaton] cost : %f (%f)\n", graph->transfer_cost, graph->transfer_cost / origin_cost);
            printf("[LearningAutomaton] totalcost / budget : %f / %f\n", graph->movecost + graph->transfer_cost, Budget);
            printf("[LearningAutomaton] iteration use : %f s\n", iteration_use_overhead);
            // cout << cnt << endl;
            log_file << graph->transfer_time << '\t' << graph->transfer_cost << '\t' << graph->movecost << endl;
        }

        auto train_end = chrono::steady_clock::now();
        graph->print();
        // printf("[LeanrningAutomaton] training use : %f s\n", (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000);
        printf("[LeanrningAutomaton] training use : %f s\n", overhead_use);
        // log_file << (double)std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_begin).count() / 1000 << endl;
        log_file << overhead_use << endl;
    }
    void check_mirror()
    {
        // #pragma omp parallel for
        for(int i = 0; i < graph->vertex_num; i++)
        {
            int ssum = 0;
            for(int j = 0; j < graph->DC_num; j++)
                ssum += graph->mirror[j][i].in_use ? 1 : 0;
            if(ssum != graph->vertex[i].num_mirror)
            {
                
                cout << ssum << " " << graph->vertex[i].num_mirror << endl;
                exit(666);
            }
            // if(graph->mirror[j][i].in_use && (graph->mirror[j][i].local_in_degree == 0 && graph->mirror[j][i].local_out_degree == 0))
            // // if(graph->mirror[j][i].local_in_degree < 0 || graph->mirror[j][i].local_out_degree < 0)
            //     {cout << j <<" " << graph->mirror[j][i].id << " " << graph->mirror[j][i].local_in_degree << " " << graph->mirror[j][i].local_in_degree << endl;
            //     exit(666);}
        
        }
    }
};

#endif