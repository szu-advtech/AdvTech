#pragma once
#include <iostream>
#include <vector>
#include <unistd.h>
using namespace std;

class Modified_Prim
{
private:
	vector<vector<int>> G;
	vector<vector<double>> recreationG;
	void Prim(int v)
	{
		// for test
		if (0)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
					cout << G[i][j] << " ";
				cout << endl;
			}
		}

		// init
		for (int i = 0; i < n; ++i)
		{
			//	dist[i] = G[0][i];
			set[i] = i;
			l[i] = G[0][i];
			parent[i] = 0;
		}
		parent[0] = -1;
		p[0] = 0;

		// prim
		for (int i = 0; i < n - 1 ; ++i)
		{
			// cout << "i = " << i << endl;
			int min_dist = INT_MAX;
			int choose = 0;
			// ����С�ı� 
			for (int j = 0; j < n; ++j)
				if (set[j] != 0 and l[j] < min_dist)
				{
					min_dist = l[j];
					choose = j;
				}
			// ���û�ҵ���˵���������
			if (choose == 0)
				break;
			// ȷ��˭Ҫ��ӽ�����
			set[choose] = 0;
			// cout << "choose " << choose << " " << parent[choose] << " " << p[parent[choose]] << endl;
			p[choose] = p[parent[choose]] + recreationG[parent[choose]][choose];

			//cout << "choose: " << choose << endl;

			// ����parent
			for (int j = 0; j < n; ++j)
				if (set[j] != 0 and G[choose][j] < l[j] and G[choose][j] != -1 and p[choose] + recreationG[choose][j] <= maxRecreationConstraint)
				{
					l[j] = G[choose][j];
					parent[j] = choose;
				}

			// ���¼����ڵ�Ԫ��
			for (int j = 1; j < n; ++j)
			{
				// j = 0 ָ����������ڵ�, ������ڵ��������
				// j ������ choose �� neighbor ��G��
				// Ҳ����choose�ɴ�j
				if (set[j] == 0 and l[j] > G[choose][j] and G[choose][j] != -1 and p[choose] + recreationG[choose][j] <= p[j])
				{
					// ���ȼ���Ƿ�ṹ����ѭ��
					vector<int> path;
					path.emplace_back(j);
					int curP = parent[choose], curV = choose;
					bool find = false;
					while (1)
					{
						curP = parent[curV];
						if (curP == j)
						{
							find = true;
							break;
						}
						if (curP == 0)
							break;
						path.emplace_back(curP);
						curV = curP;
					}
					if (!find)
					{
						l[j] = G[choose][j];
						parent[j] = choose;
						p[j] = p[choose] + recreationG[choose][j];
					}
					vector<int>().swap(path);
				}
			}
		}
		if (0)
		{
			cout << " ------- final ------- " << endl;
			cout << "set:" ;
			for (int i = 0; i < n; ++i)
			{
				cout << setw(2) << setfill(' ') << set[i] << " ";
			}
			cout << endl << "dis:" ;
			for (int i = 0; i < n; ++i)
			{
				cout << setw(2) << setfill(' ') << l[i] << " ";
			}
			cout << endl << "par:" ;
			for (int i = 0; i < n; ++i)
			{
				cout << setw(2) << setfill(' ') << parent[i] << " ";
			}
			cout << endl;
		}

		//cout << "here" << endl;
		for (int i = 1; i < n; ++i)
			total_cost += G[parent[i]][i];

		// cout << "total_cost = " << total_cost << endl;
		// cout << "---------------- End ----------------" << endl;

		vector<int>(0).swap(l);
		vector<int>(0).swap(set);
	}
public:
	int n;               // �汾��Ŀ
	int total_cost;      // �������Ļ�ԭ����
	double maxRecreationConstraint;  // ���ԭ�ɱ�������
	vector<int> parent;  // ��ǰ�汾�ĸ��ڵ�
	vector<int> l;       // ��ǰ�汾�Ļ�ԭ����
	vector<int> set;     // ��ǰ�汾���ڼ���
	vector<double> p;    // ÿ���汾�Ļ��ݳɱ�

	Modified_Prim(vector<vector<int>> Graph, vector<vector<double>> _recreationG)
	{
		n = Graph.size();
		this->G = Graph;
		this->recreationG = _recreationG;
		parent.resize(n, 0);
		l.resize(n, 0);
		set.resize(n, 0);
		p.resize(n, DBL_MAX);
		total_cost = 0;
	}

	void cal_Prim(double constraint)
	{
		this->maxRecreationConstraint = constraint;
		this->Prim(0);
	}

	int Get_total_cost()
	{
		return total_cost;
	}
};
