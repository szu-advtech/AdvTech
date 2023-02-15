#pragma once
#include <iostream>
#include <vector>
#include <unistd.h>
using namespace std;

class SPT{
    private:
        vector<vector<double>> G;
        void shortest_path_tree(int rootIndex)
        {
            // init
            dist[rootIndex] = 0, set[rootIndex] = true;
            for (int i = 0; i < n; ++i)
            {
                if (i == rootIndex)
                    continue;
                if (!(fabs(G[rootIndex][i] - (-1)) <= 1e-6))
                {
                    dist[i] = G[rootIndex][i];
                    parents[i] = rootIndex;
                }
            }
            // start
            for (int i = 0; i < n - 1; ++i)
            {
                // cout << "i = " << i << endl;
                double minDis = DBL_MAX;
                double candidate = -1;
                for (int j = 0; j < n; ++j)
                {
                    if (set[j])  // 当前节点已经添加过了 
                        continue;
                    if (minDis > dist[j])
                    {
                        candidate = j, minDis = dist[j];
                    }
                }
                // cout << "candidate = " << candidate << endl;
                // 将候选者添加进来
                set[candidate] = true;
                // 更新
                for (int j = 0; j < n; ++j)
                {
                    if (set[j])
                        continue;
                    if (!(fabs(G[candidate][j] - (-1)) <= 1e-6) and dist[candidate] + G[candidate][j] < dist[j])
                    {
                        // cout << "j = " << j << "ori = " << dist[j] << "  after = " << dist[candidate] + G[candidate][j] << endl;
                        dist[j] = dist[candidate] + G[candidate][j];
                        parents[j] = candidate;
                    }
                }
            }
            // end
        }
    public:
        int n;                  // 版本总数
        vector<int> parents;     // 记录各自的父节点
        vector<bool> set;        // 是否已加入集合
        vector<double> dist;        // 各版本距离根节点的最短路径
        SPT(vector<vector<double>> _G)
        {
            G = _G;
            this->n = G.size();
            this->parents.resize(n, -1);
            this->set.resize(n, false);
            this->dist.resize(n, DBL_MAX);
        }
        void cal_SPT(int rootIIndex)
        {
            this->shortest_path_tree(rootIIndex);
        }
};