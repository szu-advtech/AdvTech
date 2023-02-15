#include <bits/stdc++.h>
#include <unordered_set>
#include <fstream>
#include <cstring>
#include <sstream>
#include <sys/io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <dirent.h>
#include "MCA.h"
#include "Modified Prim.h"
#include "SPT.h"
using namespace std;

// 保存版本信息
struct DataInfo
{
	string filename;
	int parent_index;	 // 该节点在生成树中对应的父节点的下标
	int ori_parents[3];	 // 在原来的version graph的关系中, 原本的父节点都有谁
	int ori_parents_num; // 在原来的version graph的关系中, 原本的父节点的数目(仿真中我们定义最多有3个)
	// (-1表示当前节点实例化，无父节点)
	int linkNum;			// 主要用于skiplink里
	int restoreTime;		// 记录恢复【成本】
	int singleRestoreTime;	// 记录从其父节点恢复至当前节点的时间
	int singleStoreSize;	// 记录从其父节点恢复至当前节点的增量文件大小(如果当前版本实例化则是实例化文件的大小)
	bool constructShortcut; // 记录是否已建立shortcut
	DataInfo(string _filename) : filename(_filename)
	{
		parent_index = -1;
		linkNum = 0;
		singleRestoreTime = 0;
		singleStoreSize = 0;
		constructShortcut = false;
		restoreTime = 0;
	};
};

// 一些全局变量
const string path = "hdfeos"; // 读取文件夹的路径
vector<DataInfo> AllData; // 保存所有的版本信息
vector<vector<int>> graph;
vector<vector<double>> recreation_graph;

// 结果保存
struct ResultInfo
{
	int storage_cost;
	int recreation_cost;
	int max_recreation_cost;
	double overhead;
	ResultInfo(int _s, int _r, int _mr, double _o) : storage_cost(_s), recreation_cost(_r), max_recreation_cost(_mr), overhead(_o) {}
};
vector<ResultInfo> LMG_Result, MP_Result, LAST_Result, GitH_Result;

/**************** 通用函数 ****************/
// 返回路径下的所有文件名称
vector<string> getFiles(string path);

// 获取文件大小
// index是filename的索引下标
// 若查找 a_to_b.xdelta 文件, 则下标index 为 a 的下标
int getFileSize(string filename, int index);

// 对于下标为index的版本, 返回当前版本的父节点恢复至当前节点的时间
// 也就是只经过一层, 而不是原本的整条还原路径
int singleRecoverTime(int index);

// 返回下标为index的版本所需要的时间(沿着整条恢复路径)
int recoverVersion(int index);

// 对于下标为index的版本, 若它非实例化, 则返回生成对应的delta文件需要的时间
double produceDeltaTime(int index);

// 查找第i个AllData的所有祖先
vector<int> GetParent(int index);

// 读取最大maxRecreationCost(给MP使用)
vector<double> readMaxRecreationCost();

/**************** 仿真函数 ****************/
/* 事先将存储矩阵以及恢复成本矩阵构建出来
 * 值得注意3点:
 * 1. 是否对称
 * 2. recreation_{i, j} 是否等于 storage_{i, j}
 * 3. 每10个版本拥有彼此之间的信息(所以不会是像我们以前那样把一个超大的完全图构建出来)
 * return: 构建矩阵所需要的时间(即生成所有增量文件所需要的时间)
 */
double createMatrix(bool IsSymmetrical, bool IsEqual);

void onlyMCA();

void onlySPT();

ResultInfo LocalMoveGreedy_Scheme(int storageConstraint);

ResultInfo Prim_Scheme_Global(double maxRecreationConstraint, int storageConstraint);

ResultInfo LAST_Scheme(int storageConstraint, double alpha);

ResultInfo GitH_Scheme(int window_size, int max_depth);

void printDatasetInfo();

// 测试函数(无用途)
void test();

int main()
{
	// get filenames
	vector<string> filenames = getFiles(path);

	// init the record
	for (int i = 0; i < filenames.size(); ++i)
	{
		DataInfo obj(filenames[i]);
		AllData.emplace_back(obj);
		// cout << filenames[i] << endl;
		// if (i == 399)
		// 	break;
	}

	// resize矩阵, 这个时候就读, 后面的所有函数都能用
	int n = AllData.size();
	graph.resize(n + 1), recreation_graph.resize(n + 1);
	for (int i = 0; i < n + 1; ++i)
	{
		graph[i].resize(n + 1, -1);
		recreation_graph[i].resize(n + 1, -1);
	}

	// 读取矩阵的数据
	/* 根据后续实验, 存在非对称以及对称矩阵
	 * 所以这里需要设置bool变量以实现两种效果
	 * 此外, 原文并不是维护完全图, 而是10个版本为一组进行测试
	 * 所以这里我们依照原文, 模拟在有限信息下如何构建存储布局
	 */
	bool IsSymmetrical = false, IsEqual = false;
	double data_processing_time = 0;

	data_processing_time = createMatrix(IsSymmetrical, IsEqual);
	cout << "Create Delta Time = " << data_processing_time << "s" << endl;

	if (0)
	{
		// 输出信息
		printDatasetInfo();
		return 0;
	}

	// 统计压缩前的存储成本
	int total_size = 0;
	for (int i = 0; i < AllData.size(); ++i)
		total_size += getFileSize(AllData[i].filename, i);
	// cout << "total size = " << total_size << endl;

	// 计算一下理论最小值
	MCA mca(graph);
	mca.calMCA();
	int min_total_size = mca.Get_total_cost();
	// int min_total_size = 0;

	cout << "min = " << min_total_size << "  max = " << total_size << endl;
	// return 0;
	//  根据最大值和最小值去设置storageConstraint
	int testNum = 10;
	int gap = (total_size - min_total_size) / (1 * testNum);
	vector<int> storageConstraint;
	for (int i = 0; i < testNum and i < 1; ++i)
		storageConstraint.emplace_back(min_total_size + i * gap);

	if (0)
	{
		for (int storage : storageConstraint)
			LMG_Result.emplace_back(LocalMoveGreedy_Scheme(storage));
		// 写入结果
		ofstream fout("result/" + path + "/LMG.csv", ios::out);
		fout << "Storage limitation"
			 << ","
			 << "Storage Cost"
			 << ","
			 << "Sum of Recreation Cost"
			 << ","
			 << "Max Recreation"
			 << ","
			 << "overhead"
			 << "," << endl;
		int index = 0;
		for (ResultInfo obj : LMG_Result)
		{
			fout << storageConstraint[index] << "," << obj.storage_cost << "," << obj.recreation_cost << "," << obj.max_recreation_cost << "," << obj.overhead << endl;
			++index;
		}
		fout.close();
	}

	if (0)
	{
		vector<double> maxRecreation = readMaxRecreationCost();
		// check value
		if (maxRecreation.size() != storageConstraint.size())
		{
			// cout << "hello?" << endl;
			// 如果大小不一样, 则直接初始化为最大值
			maxRecreation.resize(storageConstraint.size());
			for (int i = 0; i < maxRecreation.size(); ++i)
				maxRecreation[i] = DBL_MAX;
		}
		for (int i = 0; i < storageConstraint.size(); ++i)
		{
			MP_Result.emplace_back(Prim_Scheme_Global(maxRecreation[i], storageConstraint[i]));
		}
		// MP_Result.emplace_back(Prim_Scheme_Global(DBL_MAX, storageConstraint[i]));
		// 写入结果
		ofstream fout("result/" + path + "/MP.csv", ios::out);
		fout << "Storage limitation"
			 << ","
			 << "Storage Cost"
			 << ","
			 << "Sum of Recreation Cost"
			 << ","
			 << "Max Recreation"
			 << ","
			 << "overhead"
			 << "," << endl;
		int index = 0;
		for (ResultInfo obj : MP_Result)
		{
			fout << storageConstraint[index] << "," << obj.storage_cost << "," << obj.recreation_cost << "," << obj.max_recreation_cost << "," << obj.overhead << endl;
			++index;
		}
		fout.close();
	}

	if (0)
	{
		for (int storage : storageConstraint)
			LAST_Result.emplace_back(LAST_Scheme(storage, 0.9));
		// 写入结果
		ofstream fout("result/" + path + "/LAST.csv", ios::out);
		fout << "Storage limitation"
			 << ","
			 << "Storage Cost"
			 << ","
			 << "Sum of Recreation Cost"
			 << ","
			 << "Max Recreation"
			 << ","
			 << "overhead"
			 << "," << endl;
		int index = 0;
		for (ResultInfo obj : LAST_Result)
		{
			fout << storageConstraint[index] << "," << obj.storage_cost << "," << obj.recreation_cost << "," << obj.max_recreation_cost << "," << obj.overhead << endl;
			++index;
		}
		fout.close();
	}

	if (0)
	{
		// vector<int> windowSizes = {50, 25, 20, 10};
		vector<int> windowSizes = {500, 250, 125, 100};
		for (int window : windowSizes)
			GitH_Result.emplace_back(GitH_Scheme(window, 50));
		// 写入结果
		ofstream fout("result/" + path + "/GitH.csv", ios::out);
		fout << "Storage Cost"
			 << ","
			 << "Sum of Recreation Cost"
			 << ","
			 << "Max Recreation"
			 << ","
			 << "overhead"
			 << "," << endl;
		for (ResultInfo obj : GitH_Result)
		{
			fout << obj.storage_cost << "," << obj.recreation_cost << "," << obj.max_recreation_cost << "," << obj.overhead << endl;
		}
		fout.close();
	}

	// 保存结果
	if (1)
	{
		// 填写信息
		ofstream fout("debug result/" + path + "_result.csv", ios::out);
		fout << "version id"
			 << ","
			 << "singleStoreSize"
			 << ","
			 << "singleRestoreTime"
			 << ","
			 << "restoreTime"
			 << ","
			 << "parent"
			 << ","
			 << "parent name"
			 << ","
			 << "child name" << endl;
		for (int i = 0; i < AllData.size(); ++i)
		{
			fout << i << "," << AllData[i].singleStoreSize << "," << AllData[i].singleRestoreTime << "," << AllData[i].restoreTime << ","
				 << AllData[i].parent_index << ",";
			if (AllData[i].parent_index != -1)
				fout << AllData[AllData[i].parent_index].filename << "," << AllData[i].filename;
			fout << endl;
		}

		fout.close();
	}

	return 0;
}

vector<string> getFiles(string path)
{
	vector<string> ans;

	DIR *d = opendir(path.c_str());
	if (d == nullptr)
	{
		perror("opendir error.");
		exit(1);
	}
	struct dirent *entry;
	while ((entry = readdir(d)) != nullptr)
	{
		string file = entry->d_name;
		if (strcmp(file.c_str(), ".") != 0 and strcmp(file.c_str(), "..") != 0 and strcmp(file.c_str(), "Delta Size") != 0 and strcmp(file.c_str(), "Produce Delta Time") != 0 and strcmp(file.c_str(), "Recover Time") != 0 and strcmp(file.c_str(), "Recreation Cost") != 0 and strstr(file.c_str(), "xdelta") == nullptr and strstr(file.c_str(), "bat") == nullptr and strstr(file.c_str(), "txt") == nullptr and strstr(file.c_str(), "-ver") == nullptr)
			ans.emplace_back(file);
	}

	return ans;
}

int getFileSize(string filename, int index)
{
	int size = 0;
	// 如果是实打实的文件
	if (strstr(filename.c_str(), ".xdelta") == NULL)
	{
		string end = path + "/" + filename;
		std::ifstream in(end.c_str());
		in.seekg(0, std::ios::end);
		size = in.tellg();
		in.close();
		// return size; //单位是：Byte
		return size / 1024; // KB
	}
	else
	{
		// 读取记录文件
		string file = AllData[index].filename;
		ifstream fin;
		fin.open(path + "/Delta Size/" + file.substr(0, file.find('.')) + ".txt");
		string buf;
		string deltaName, deltaSize;
		stringstream ss;
		while (getline(fin, buf))
		{
			ss << buf;
			ss >> deltaName >> deltaSize;
			if (filename == deltaName)
			{
				size = atoi(deltaSize.c_str());
				ss.clear();
				break;
			}
			ss.clear();
		}
		fin.close();
		return size; // 注意存的时候已经是 KB 为单位的了
	}
}

int singleRecoverTime(int index)
{
	int p = AllData[index].parent_index;
	if (p == -1)
	{
		// 实例化版本
		// cout << AllData[index].filename << "has been materialized!" << endl;
		return getFileSize(AllData[index].filename, index);
	}
	else
	{
		string file = AllData[p].filename;
		int recoverTime = 0;
		string fileName = AllData[p].filename + "_to_" + AllData[index].filename + ".xdelta";
		ifstream fin;
		fin.open(path + "/Recreation Cost/" + file.substr(0, file.find('.')) + ".txt");
		string buf;
		string deltaName, deltaTime;
		stringstream ss;
		while (getline(fin, buf))
		{
			ss << buf;
			ss >> deltaName >> deltaTime;
			if (fileName == deltaName)
			{
				// find
				recoverTime = atoi(deltaTime.c_str());
				ss.clear();
				break;
			}
			ss.clear();
		}
		fin.close();
		return recoverTime;
	}
}

int recoverVersion(int index)
{
	int totalRecoverTime = 0;
	while (index != -1)
	{
		// cout << index << " <- " << AllData[index].parent_index << endl;
		totalRecoverTime += AllData[index].singleRestoreTime;
		index = AllData[index].parent_index;
	}
	return totalRecoverTime;
}

double produceDeltaTime(int index)
{
	int p = AllData[index].parent_index;
	if (p == -1)
	{
		// 实例化版本
		return 0;
	}
	else
	{
		string file = AllData[p].filename;
		double produceTime = 0;
		string fileName = AllData[p].filename + "_to_" + AllData[index].filename + ".xdelta";
		ifstream fin;
		fin.open(path + "/Produce Delta Time/" + file.substr(0, file.find('.')) + ".txt");
		string buf;
		string deltaName, deltaTime;
		stringstream ss;
		while (getline(fin, buf))
		{
			ss << buf;
			ss >> deltaName >> deltaTime;
			if (fileName == deltaName)
			{
				// find
				produceTime = atof(deltaTime.c_str());
				ss.clear();
				break;
			}
			ss.clear();
		}
		fin.close();
		return produceTime;
	}
}

vector<int> GetParent(int index)
{
	vector<int> parent;
	if (index == -1)
		return parent;
	int p = AllData[index].parent_index;
	while (p != -1)
	{
		parent.push_back(p);
		p = AllData[p].parent_index;
	}
	return parent;
}

double createMatrix(bool IsSymmetrical, bool IsEqual)
{
	double data_processing_time = 0;
	int n = AllData.size();
	if (IsSymmetrical)
	{
		// 对称矩阵
		for (int i = 0; i < n + 1; ++i)
		{
			cout << "i = " << i << endl;
			for (int j = i; j < n + 1; ++j) // 注意是对称矩阵
			{
				if (i == j or j == 0)
					continue;
				else if (i == 0) // 虚拟节点
				{
					graph[i][j] = getFileSize(AllData[j - 1].filename, j - 1);
					AllData[j - 1].parent_index = -1;
				}
				else
				{
					string deltaName = AllData[i - 1].filename + "_to_" + AllData[j - 1].filename + ".xdelta";
					graph[i][j] = getFileSize(deltaName, i - 1);
					graph[j][i] = graph[i][j]; // 对称矩阵
					AllData[j - 1].parent_index = i - 1;
				}
				if (IsEqual)
				{
					recreation_graph[i][j] = graph[i][j];
				}
				else
				{
					recreation_graph[i][j] = singleRecoverTime(j - 1);
				}
				if (i != 0) // 如果不是虚拟节点
					recreation_graph[j][i] = recreation_graph[i][j];
				data_processing_time += produceDeltaTime(j - 1);
			}
		}
	}
	else
	{
		// 非对称矩阵
		for (int i = 0; i < n + 1; ++i)
		{
			cout << "i = " << i << endl;
			for (int j = 0; j < n + 1; ++j)
			{
				if (i == j or j == 0)
					continue;
				else if (i == 0) // 虚拟节点
				{
					graph[i][j] = getFileSize(AllData[j - 1].filename, j - 1);
					AllData[j - 1].parent_index = -1;
				}
				else
				{
					string deltaName = AllData[i - 1].filename + "_to_" + AllData[j - 1].filename + ".xdelta";
					graph[i][j] = getFileSize(deltaName, i - 1);
					AllData[j - 1].parent_index = i - 1;
				}

				if (IsEqual)
					recreation_graph[i][j] = graph[i][j];
				else
					recreation_graph[i][j] = singleRecoverTime(j - 1);
				data_processing_time += produceDeltaTime(j - 1);
			}
		}
	}
	// 在这里调整矩阵
	// 保证每10个版本拥有彼此之间的增量信息
	bool adjust = true;
	if (adjust)
	{
		for (int i = 1; i <= n; i += 10)
		{
			int start = i, end = i + 9;
			for (int row = i; row < i + 10 and row <= n; ++row)
			{
				// 前
				for (int col = 1; col < start; ++col)
				{
					graph[row][col] = -1;
					recreation_graph[row][col] = -1;
					AllData[col - 1].parent_index = row - 1;
					data_processing_time -= produceDeltaTime(col - 1); // 因为之前统计多了, 所以减掉
				}
				// 后
				for (int col = end + 1; col <= n; ++col)
				{
					if (row == (i + 9) and col == (end + 1))
						continue; // 衔接
					graph[row][col] = -1;
					recreation_graph[row][col] = -1;
					AllData[col - 1].parent_index = row - 1;
					if (!IsSymmetrical)
						data_processing_time -= produceDeltaTime(col - 1); // 因为之前统计多了, 所以减掉
				}
			}
		}
	}

	return data_processing_time;
}

void onlyMCA()
{
	int n = AllData.size();
	double overhead = 0;
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// MCA
	MCA mca(graph);
	mca.calMCA();

	if (1)
		cout << "MAC finish." << endl;

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 更新对应的关系
	for (int i = 1; i < n + 1; ++i)
	{
		if (mca.parent[i] - 1 == -1)
		{
			AllData[i - 1].parent_index = -1;
			AllData[i - 1].singleStoreSize = getFileSize(AllData[i - 1].filename, i - 1);
		}
		else
		{
			AllData[i - 1].parent_index = mca.parent[i] - 1;
			string fileName = AllData[AllData[i - 1].parent_index].filename + "_to_" + AllData[i - 1].filename + ".xdelta";
			AllData[i - 1].singleStoreSize = getFileSize(fileName, AllData[i - 1].parent_index);
		}
		AllData[i - 1].singleRestoreTime = singleRecoverTime(i - 1);
		total_cost += AllData[i - 1].singleStoreSize;
	}

	// 输出
	total_cost = 0, total_time = 0;

	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- MCA --------" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize: " << materializedVersionSize << "KB" << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;
}

void onlySPT()
{
	int n = AllData.size();
	double overhead = 0;
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// SPT
	SPT spt(recreation_graph);
	spt.cal_SPT(0);

	if (1)
		cout << "MAC finish." << endl;

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 更新对应的关系
	for (int i = 1; i < n + 1; ++i)
	{
		if (spt.parents[i] - 1 == -1)
		{
			AllData[i - 1].parent_index = -1;
			AllData[i - 1].singleStoreSize = getFileSize(AllData[i - 1].filename, i - 1);
		}
		else
		{
			AllData[i - 1].parent_index = spt.parents[i] - 1;
			string fileName = AllData[AllData[i - 1].parent_index].filename + "_to_" + AllData[i - 1].filename + ".xdelta";
			AllData[i - 1].singleStoreSize = getFileSize(fileName, AllData[i - 1].parent_index);
		}
		AllData[i - 1].singleRestoreTime = singleRecoverTime(i - 1);
		total_cost += AllData[i - 1].singleStoreSize;
	}

	// 输出
	total_cost = 0, total_time = 0;

	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- SPT --------" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize: " << materializedVersionSize << "KB" << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;
}

ResultInfo LocalMoveGreedy_Scheme(int storageConstraint)
{
	int n = AllData.size();
	double overhead = 0;
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;
	// 先执行最小树形图

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// MCA
	MCA mca(graph);
	mca.calMCA();

	if (1)
		cout << "MAC finish." << endl;

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	cout << "MCA time = " << overhead << "s" << endl;
	// cout << "here storage Cost = " << mca.Get_total_cost() << endl;

	// SPT
	gettimeofday(&t1, NULL);
	SPT spt(recreation_graph);
	spt.cal_SPT(0);

	if (1)
	{
		cout << "SPT finish." << endl;
	}

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 更新对应的关系
	for (int i = 1; i < n + 1; ++i)
	{
		AllData[i - 1].parent_index = mca.parent[i] - 1;
		AllData[i - 1].singleStoreSize = graph[mca.parent[i]][i];
		AllData[i - 1].singleRestoreTime = recreation_graph[mca.parent[i]][i];
		total_cost += AllData[i - 1].singleStoreSize;
		// cout << i - 1 << " " << spt.parents[i] - 1 << " vs " << mca.parent[i] - 1 << endl;
	}

	storageConstraint = storageConstraint >= total_cost ? storageConstraint : total_cost;

	// cout << "check here total cost = " << total_cost << "  and storageConstraint = " << storageConstraint << endl;

	// spt的对应关系
	struct relation
	{
		int index;
		int parent;
		int storageInc;
		double p;
		relation(int _i, int _p) : index(_i), parent(_p)
		{
			storageInc = 0;
			p = 0;
		}
	};

	gettimeofday(&t1, NULL);

	vector<relation> relations;
	for (int i = 1; i < n + 1; ++i)
	{
		// 主要是找不存在于MCA的边
		if (spt.parents[i] != mca.parent[i])
		{
			if (0) // 新的写法
			{
				// 存入
				relation obj(i - 1, spt.parents[i] - 1);
				// 开始计算 storageInc, p1, p2, p
				int deltaStorage = 0;
				int deltaRecreation = recreation_graph[spt.parents[i]][i];

				// int recreationDec = recreation_graph[AllData[i - 1].parent_index + 1][i] - deltaRecreation;  // 这个不对
				// 因为公式里面计算的是 reduction in the sum of recreation costs
				// 所以是需要统计两次的recreation cost 从而计算出下降量
				// 1. 先保存原有的信息
				int ori_parent = AllData[i - 1].parent_index;
				int ori_singleStoreSize = AllData[i - 1].singleStoreSize;
				int ori_singleRestoreTime = AllData[i - 1].singleRestoreTime;
				// 2. 计算现在的recreationCost
				int cur_recreation = recoverVersion(i - 1);
				// 3. 计算新的recreationCost
				// 先检查spt.parents[i] - 1的祖先有没有i - 1, 避免死循环
				vector<int> path;
				if (spt.parents[i] != 0)
					path = GetParent(spt.parents[i] - 1);
				unordered_set<int> parents(path.begin(), path.end());
				// 如果找到了, 那必会死循环, 直接实例化吧
				if (parents.find(i - 1) != parents.end())
				{
					spt.parents[i] = 0;
					obj.parent = -1;
				}

				AllData[i - 1].parent_index = spt.parents[i] - 1;
				// AllData[i - 1].singleRestoreTime = singleRecoverTime(i - 1);
				AllData[i - 1].singleRestoreTime = recreation_graph[spt.parents[i]][i];
				int new_recreation = recoverVersion(i - 1);
				// 4. 计算差异
				int recreationDec = cur_recreation - new_recreation;
				// 5. 还原
				AllData[i - 1].parent_index = ori_parent;
				AllData[i - 1].singleStoreSize = ori_singleStoreSize;
				AllData[i - 1].singleRestoreTime = ori_singleRestoreTime;

				// 计算storageDec
				if (spt.parents[i] - 1 == -1)
				{
					deltaStorage = getFileSize(AllData[i - 1].filename, i - 1);
				}
				else
				{
					string deltaName = AllData[spt.parents[i] - 1].filename + "_to_" + AllData[i - 1].filename + ".xdelta";
					deltaStorage = getFileSize(deltaName, spt.parents[i] - 1);
				}
				int storageInc = deltaStorage - AllData[i - 1].singleStoreSize;
				obj.storageInc = storageInc;
				if (storageInc == 0)
					storageInc = 0.1;

				if (0)
				{
					// 所以要遍历所有版本 如果它的还原路径里有 i-1 号版本, 则我们就要记录
					int num = 1;
					for (int index = 0; index < n; ++index)
					{
						if (index == i - 1)
							continue;
						vector<int> path = GetParent(index);
						bool find = false;
						for (int j = 0; j < path.size(); ++j)
							if (path[j] == index)
							{
								find = true;
								break;
							}
						if (find)
							++num;
						vector<int>().swap(path);
					}
					double p = 1.0 * recreationDec * num / storageInc;
					obj.p = p;
					relations.emplace_back(obj);
				}
				double p = 1.0 * recreationDec / storageInc;
				obj.p = p;
				relations.emplace_back(obj);
			}
			else
			{
				// 避免构成死循环
				if (spt.parents[i] - 1 != -1)
				{
					vector<int> parents = GetParent(spt.parents[i] - 1);
					unordered_set<int> set(parents.begin(), parents.end());
					if (set.find(i - 1) != set.end())
						spt.parents[i] = 0;
				}

				// 存入
				relation obj(i - 1, spt.parents[i] - 1);
				// 开始计算 storageInc, p1, p2, p
				int deltaStorage = 0;
				// 保存原来的父节点
				int ori_parentID = AllData[i - 1].parent_index;
				double ori_singleRestoreTime = AllData[i - 1].singleRestoreTime;
				// 这里修改找一下recreation
				AllData[i - 1].parent_index = spt.parents[i] - 1;
				// AllData[i - 1].singleRestoreTime = singleRecoverTime(i - 1);
				AllData[i - 1].singleRestoreTime = recreation_graph[spt.parents[i]][i];
				double deltaRecreation = recoverVersion(i - 1);
				// 还原回去
				AllData[i - 1].parent_index = ori_parentID;
				AllData[i - 1].singleRestoreTime = ori_singleRestoreTime;
				deltaStorage = graph[spt.parents[i]][i];
				int storageInc = deltaStorage - AllData[i - 1].singleStoreSize;
				obj.storageInc = storageInc;
				if (storageInc == 0)
					storageInc = 0.1;

				// double recreationDec = recreation_graph[AllData[i - 1].parent_index + 1][i] - deltaRecreation;
				double recreationDec = recoverVersion(i - 1) - deltaRecreation;
				// 因为公式里面计算的是 reduction in the sum of recreation costs
				// 所以要遍历所有版本 如果它的还原路径里有 i-1 号版本, 则我们就要记录
				int num = 1;
				for (int index = 0; index < n; ++index)
				{
					if (index == i - 1)
						continue;
					vector<int> path = GetParent(index);
					unordered_set<int> set(path.begin(), path.end());
					if (set.find(i - 1) != set.end())
						++num;
					vector<int>().swap(path);
				}
				double p = recreationDec * num / storageInc;
				obj.p = p;
				relations.emplace_back(obj);
			}
		}
	}

	// 排序
	sort(relations.begin(), relations.end(), [](relation &a, relation &b)
		 { return a.p > b.p; });


	// 依次添加
	for (int i = 0; i < relations.size(); ++i)
	{
		// cout << total_cost << " + " << relations[i].storageInc << " = " << total_cost + relations[i].storageInc << " vs " << storageConstraint << endl;
		if (total_cost + relations[i].storageInc > storageConstraint)
			break;
		int index = relations[i].index, parentIndex = relations[i].parent;
		AllData[index].parent_index = parentIndex;
		AllData[index].singleStoreSize += relations[i].storageInc;
		// AllData[index].singleRestoreTime = singleRecoverTime(index);
		AllData[index].singleRestoreTime = recreation_graph[parentIndex + 1][index + 1];
		total_cost += relations[i].storageInc;
	}

	if (0)
	{
		// for test
		cout << "after add the total_cost = " << total_cost << endl;
	}

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 输出
	total_cost = 0, total_time = 0;

	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- Local Move Greedy Scheme --------" << endl;
	cout << "constraint = " << storageConstraint << "KB" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize: " << materializedVersionSize << "KB" << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;

	return {total_cost, total_time, maxRecreationCost, overhead};
}

ResultInfo Prim_Scheme_Global(double maxRecreationConstraint, int storageConstraint)
{
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;
	int n = AllData.size();
	double overhead = 0;

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// Modified_Prim tree(graph, restore_graph);
	Modified_Prim tree(graph, recreation_graph);
	tree.cal_Prim(maxRecreationConstraint);

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	for (int i = 1; i < n + 1; ++i)
	{
		AllData[i - 1].parent_index = tree.parent[i] - 1;
		AllData[i - 1].singleStoreSize = graph[tree.parent[i]][i];
		AllData[i - 1].singleRestoreTime = recreation_graph[tree.parent[i]][i];
	}

	// 统计当前的存储成本
	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		AllData[i].restoreTime = recoverVersion(i);
	}
	storageConstraint = storageConstraint >= total_cost ? storageConstraint : total_cost;

	// cout << total_cost << " vs " << storageConstraint << endl;
	if (0 and total_cost < storageConstraint)
	{
		//	cout << "here!" << endl;
		// 如果还有可用空间, 开始考虑实例化recreation cost最大的那些版本
		struct Info
		{
			int index;
			int storageInc;
			int recreationCost;
			Info(int _i, int _s, int _t) : index(_i), storageInc(_s), recreationCost(_t){};
		};
		vector<Info> record;
		// 记录
		for (int i = 0; i < AllData.size(); ++i)
		{
			int storageInc = getFileSize(AllData[i].filename, i) - AllData[i].singleStoreSize;
			record.emplace_back(Info(i, storageInc, AllData[i].restoreTime));
		}
		// 排序
		sort(record.begin(), record.end(), [](Info &a, Info &b)
			 { return a.recreationCost > b.recreationCost; });

		for (int i = 0; i < record.size(); ++i)
		{
			//	cout << total_cost << " + " << record[i].storageInc << " = " << total_cost + record[i].storageInc <<  " vs " << storageConstraint << endl;
			if (total_cost + record[i].storageInc > storageConstraint)
				break;
			total_cost += record[i].storageInc;
			AllData[record[i].index].parent_index = -1;
			AllData[record[i].index].singleStoreSize += record[i].storageInc;
			AllData[record[i].index].singleRestoreTime = singleRecoverTime(record[i].index);
		}
	}

	// 接下来才是真正地统计数据
	total_cost = 0, total_time = 0, materializedVersionSize = 0;
	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- Prim Scheme --------" << endl;
	cout << "constraint = " << storageConstraint << "KB" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize:" << materializedVersionSize << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost_Constraint:" << maxRecreationConstraint << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;

	return {total_cost, total_time, maxRecreationCost, overhead};
}

ResultInfo LAST_Scheme(int storageConstraint, double alpha)
{
	// recreation_{i, j} = storage_{i, j}
	// the matrix is symmetric
	int n = AllData.size();
	double overhead = 0;
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// MCA
	MCA mca(graph);
	mca.calMCA();

	if (1)
		cout << "MAC finish." << endl;

	// SPT
	SPT spt(recreation_graph);
	spt.cal_SPT(0);

	if (1)
	{
		cout << "SPT finish." << endl;
	}

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 更新对应的关系
	for (int i = 1; i < n + 1; ++i)
	{
		AllData[i - 1].parent_index = mca.parent[i] - 1;
		AllData[i - 1].singleStoreSize = graph[mca.parent[i]][i];
		AllData[i - 1].singleRestoreTime = recreation_graph[mca.parent[i]][i];
		total_cost += AllData[i - 1].singleStoreSize;
	}
	storageConstraint = storageConstraint >= total_cost ? storageConstraint : total_cost;

	// cout << "total_cost = " << total_cost << "  storageConstraint = " << storageConstraint << endl;

	// 统计Vi在SPT中与根节点的距离
	struct SPT_Info
	{
		int index;
		int parent;
		int dis;
		SPT_Info(int _i, int _p) : index(_i), parent(_p)
		{
			dis = 0;
		};
	};

	vector<SPT_Info> SPT_Tree;
	for (int i = 0; i < n; ++i)
	{
		SPT_Info obj(i, spt.parents[i + 1] - 1);
		int dis = 0;
		int curNode = i, curParent = spt.parents[i + 1] - 1;
		while (true)
		{
			if (curParent == -1)
			{
				dis += recreation_graph[0][curNode + 1];
			}
			else
			{
				dis += recreation_graph[curParent + 1][curNode + 1];
			}
			if (curParent == -1)
				break;
			curNode = curParent;
			curParent = spt.parents[curNode + 1] - 1;
		}
		obj.dis = dis;
		SPT_Tree.emplace_back(obj);
		// cout << "i = " << i << " 's dis = " << dis << endl;
	}

	if (0)
	{
		// for test
		for (int i = 0; i < n; ++i)
		{
			cout << SPT_Tree[i].index << " " << SPT_Tree[i].parent << " " << SPT_Tree[i].dis << endl;
		}
	}

	/* DFS 之前需要转换关系
	 * 因为我们现在是反着存储关系的(当前结点寻找父节点)
	 * 而DFS是从顶至下遍历的, 所以需要能让当前节点找到自己的孩子结点
	 */
	// DFS会用上的结构体
	struct Node
	{
		int index;
		int d;
		vector<int> children;
		Node(int _i) : index(_i)
		{
			d = 0;
		}
	};

	vector<Node> nodes;
	for (int i = 0; i < n; ++i)
	{
		Node node(i);
		for (int j = 0; j < n; ++j)
			if (AllData[j].parent_index == i)
				node.children.emplace_back(j);
		nodes.emplace_back(node);
	}
	// 访问标记
	vector<bool> IsVisited(n, false);
	// 先获取根节点, 根节点作为DFS的起点
	vector<int> root;
	for (int index = 0; index < n; ++index)
		if (AllData[index].parent_index == -1)
			root.emplace_back(index);

	if (0)
	{
		// for test
		cout << "before:" << endl;
		for (int i = 0; i < n; ++i)
			cout << i << " <- " << AllData[i].parent_index << "  SingleStore = " << AllData[i].singleStoreSize << endl;
		cout << endl;
	}

	cout << "start DFS" << endl;
	// DFS
	// 从根节点开始遍历
	for (int rootID : root)
	{
		int accD = 0;
		stack<int> st;
		st.emplace(rootID);
		while (!st.empty())
		{
			int nodeID = st.top();
			// cout << "here is " << nodeID << endl;
			if (IsVisited[nodeID] == false)
			{
				// 初次访问需要累加accD
				accD += AllData[nodeID].singleRestoreTime;
				// 更新d(即当前情况下, 节点的恢复成本)
				double curRecreation = recoverVersion(nodeID);
				nodes[nodeID].d = (int)curRecreation;
				// nodes[nodeID].d = accD;
				// 更新标记位
				IsVisited[nodeID] = true;
			}

			// 原算法的两个if
			int parent = AllData[nodeID].parent_index;

			if (parent != -1 and nodes[nodeID].d > nodes[parent].d + (int)recreation_graph[parent + 1][nodeID + 1])
			{
				nodes[nodeID].d = nodes[parent].d + (int)recreation_graph[parent + 1][nodeID + 1];
				AllData[nodeID].parent_index = parent;
				AllData[nodeID].singleStoreSize = graph[parent + 1][nodeID + 1];
				AllData[nodeID].singleRestoreTime = recreation_graph[parent + 1][nodeID + 1];
			}

			// cout << "first if" << endl;

			// 第二个if
			if (nodes[nodeID].d > (int)(alpha * SPT_Tree[nodeID].dis))
			{
				// 这个函数需要添加限制: 在不超过storage Limitation的情况下才可以修改这个值
				int curStorage = AllData[nodeID].singleStoreSize;
				int curParent = AllData[nodeID].parent_index;
				// if update
				AllData[nodeID].parent_index = SPT_Tree[nodeID].parent;
				int newStorage = graph[AllData[nodeID].parent_index + 1][nodeID + 1];
				if (newStorage == -1)
				{
					exit(0);
				}
				// 计算存储成本增加比
				int storageInc = newStorage - curStorage;
				// cout << "oriParent: " << curParent << "  nowParent: " << SPT_Tree[nodeID].parent << endl;
				// cout << "storageInc: " << storageInc << endl;
				if (storageInc + total_cost > storageConstraint)
				{
					// 不能添加
					AllData[nodeID].parent_index = curParent;
				}
				else
				{
					// cout << graph[SPT_Tree[nodeID].parent + 1][nodeID + 1] << " " << graph[curParent + 1][nodeID + 1] << endl;
					// cout << newStorage << " " << curStorage << endl;
					// cout << storageInc << " + " << total_cost << " = " << storageInc + total_cost << " vs " << storageConstraint << endl;

					AllData[nodeID].parent_index = curParent; // 因为上面修改了, 所以这里先还原一下
					// 添加之前还需要看看会不会变成死循环
					vector<int> parents = GetParent(SPT_Tree[nodeID].parent);
					// cout << "?" << endl;
					bool find = false;
					for (int parent : parents)
						if (parent == nodeID)
						{
							find = true;
							break;
						}
					if (!find)
					{
						total_cost += storageInc;
						// cout << "nodeID:" << nodeID << endl;
						nodes[nodeID].d = SPT_Tree[nodeID].dis;
						// cout << SPT_Tree[nodeID].dis << endl;
						AllData[nodeID].parent_index = SPT_Tree[nodeID].parent;
						AllData[nodeID].singleStoreSize = graph[AllData[nodeID].parent_index + 1][nodeID + 1];
						AllData[nodeID].singleRestoreTime = recreation_graph[AllData[nodeID].parent_index + 1][nodeID + 1];
						// cout << SPT_Tree[nodeID].parent << endl;
						if (SPT_Tree[nodeID].parent != -1)
							nodes[SPT_Tree[nodeID].parent].children.emplace_back(nodeID);
					}
					else
					{
						// 不能添加
						AllData[nodeID].parent_index = curParent;
					}
				}
				if (0)
				{
					// cout << "nodeID:" << nodeID << endl;
					nodes[nodeID].d = SPT_Tree[nodeID].dis;
					// cout << SPT_Tree[nodeID].dis << endl;
					AllData[nodeID].parent_index = SPT_Tree[nodeID].parent;
					// cout << SPT_Tree[nodeID].parent << endl;
					if (SPT_Tree[nodeID + 1].parent != -1)
						nodes[SPT_Tree[nodeID + 1].parent].children.emplace_back(nodeID);
				}
			}

			// 检查接下来还需不需要访问它的孩子
			bool IsFinish = true;
			for (int i = 0; i < nodes[nodeID].children.size(); ++i)
			{
				if (IsVisited[nodes[nodeID].children[i]] == false)
				{
					IsFinish = false;
					st.emplace(nodes[nodeID].children[i]);
					break;
				}
			}
			// cout << "find " << IsFinish << endl;
			if (IsFinish)
			{
				// 因为访问完了, 所以accD也要相应的减去
				accD -= AllData[nodeID].singleStoreSize;
				st.pop();
			}
			// cout << "FINISH!" << endl;
		}
	}

	// 统计当前的存储成本
	total_cost = 0;
	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		AllData[i].restoreTime = recoverVersion(i);
	}

	// 输出
	total_cost = 0, total_time = 0;

	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
		if (AllData[i].singleStoreSize == -1)
		{
			cout << i << " <- " << AllData[i].parent_index << endl;
			cout << "理论上: " << graph[AllData[i].parent_index + 1][i + 1] << endl;
			cout << AllData[i].filename << endl;
			if (AllData[i].parent_index != -1)
				cout << AllData[AllData[i].parent_index].filename << endl;
			exit(0);
		}
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- LAST Scheme --------" << endl;
	cout << "constraint = " << storageConstraint << "KB" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize: " << materializedVersionSize << "KB" << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;

	return {total_cost, total_time, maxRecreationCost, overhead};
}

ResultInfo GitH_Scheme(int window_size, int max_depth)
{
	int n = AllData.size();
	double overhead = 0;
	int total_cost = 0, total_time = 0;
	int materializedVersionSize = 0, maxRecreationCost = 0;

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// 第一个版本实例化
	AllData[0].parent_index = -1;
	AllData[0].singleStoreSize = getFileSize(AllData[0].filename, 0);
	AllData[0].singleRestoreTime = singleRecoverTime(0);

	// 后面开始就按照算法流程进行
	int left = 0, right = 0;	// 用双指针维护窗口
	for (int v = 1; v < n; ++v) // 第0号已经设定完毕了, 所以不用处理
	{
		// cout << "v = " << v << endl;
		//  从第1号version开始处理
		double min_ratio = DBL_MAX;
		int parentID = -1;
		// cout << "left = " << left << "  right = " << right << endl;
		for (int i = left; i <= right; ++i)
		{
			int curVersionDepth = GetParent(i).size();
			if (curVersionDepth == max_depth or graph[i + 1][v + 1] == -1)
				continue;
			int diffDepth = max_depth - curVersionDepth;
			int deltaSize = graph[i + 1][v + 1];
			double ratio = deltaSize * 1.0 / diffDepth;
			if (ratio < min_ratio)
			{
				min_ratio = ratio;
				parentID = i;
			}
			// cout << "diffDepth =  " << diffDepth << "  deltaSize = " << deltaSize << endl;
			// cout << "ratio = " << ratio << endl << endl;
		}
		if (parentID == -1)
		{
			// 也就是窗口内的所有结点深度都已经到了最大深度
			// 那么这个时候这个version别无他法, 只能实例化了
			AllData[v].parent_index = -1;
			AllData[v].singleStoreSize = graph[parentID + 1][v + 1];
			// AllData[v].singleStoreSize = getFileSize(AllData[v].filename, v);
			AllData[v].singleRestoreTime = recreation_graph[parentID + 1][v + 1];
			// AllData[v].singleRestoreTime = singleRecoverTime(v);
			// 同样, 当前窗口可以舍弃, 新创建一个窗口
			left = right = v;
		}
		else
		{
			// 如果能找到可以建立关系的结点
			AllData[v].parent_index = parentID;
			string deltaName = AllData[parentID].filename + "_to_" + AllData[v].filename + ".xdelta";
			AllData[v].singleStoreSize = graph[parentID + 1][v + 1];
			// AllData[v].singleStoreSize = getFileSize(deltaName, parentID);
			AllData[v].singleRestoreTime = recreation_graph[parentID + 1][v + 1];
			// AllData[v].singleRestoreTime = singleRecoverTime(v);
			++right; // 指针右移
			if (right - left + 1 > window_size)
				++left; // 如果窗口满了, 左指针更新一下
		}
	}

	gettimeofday(&t2, NULL);
	overhead += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;

	// 输出
	total_cost = 0, total_time = 0;

	for (int i = 0; i < n; ++i)
	{
		total_cost += AllData[i].singleStoreSize;
		if (AllData[i].parent_index == -1)
			materializedVersionSize += AllData[i].singleStoreSize;
	}

	for (int i = 0; i < n; ++i)
	{
		AllData[i].restoreTime = recoverVersion(i);
		total_time += AllData[i].restoreTime;
		if (AllData[i].restoreTime > maxRecreationCost)
			maxRecreationCost = AllData[i].restoreTime;
	}

	cout << "-------- GitH Scheme --------" << endl;
	cout << "total_cost:" << total_cost << "KB" << endl;
	cout << "materializedVersionSize: " << materializedVersionSize << "KB" << endl;
	cout << "total_RecreationCost:" << total_time << "KB" << endl;
	cout << "max_RecreationCost:" << maxRecreationCost << "KB" << endl;
	cout << " overhead :" << overhead << "s" << endl;

	return {total_cost, total_time, maxRecreationCost, overhead};
}

void printDatasetInfo()
{
	cout << "Number of versions = " << AllData.size() << endl;

	// delta 数目
	int deltasNum = 0;
	for (int i = 1; i < graph.size(); ++i)
		for (int j = 1; j < graph.size(); ++j)
			if (graph[i][j] != -1)
				++deltasNum;
	cout << "Number of Deltas = " << deltasNum << endl;

	// 总的存储成本
	int total_size = 0;
	for (int i = 0; i < AllData.size(); ++i)
		total_size += getFileSize(AllData[i].filename, i);
	// 平均存储成本
	double aveSize = total_size * 1.0 / AllData.size();
	cout << "Average version size(KB)" << aveSize << endl;

	// 只有最小生成树
	onlyMCA();

	// 只有最短路径树
	onlySPT();
}

void test()
{
	int n = AllData.size();
	double overhead = 0;
	double total_cost = 0, total_time = 0, materializedVersionSize = 0;
	// 先执行最小树形图
	vector<vector<int>> storage_graph(n + 1, vector<int>(n + 1, -1));
	vector<vector<double>> recreation_graph(n + 1, vector<double>(n + 1, -1));
	for (int i = 0; i < n + 1; ++i)
	{
		cout << "i = " << i << endl;
		for (int j = 0; j < n + 1; ++j)
		{
			if (i == j or j == 0)
				continue;
			else if (i == 0) // 虚拟节点
			{
				storage_graph[i][j] = getFileSize(AllData[j - 1].filename, j - 1);
				AllData[j - 1].parent_index = -1;
			}
			else
			{
				string deltaName = AllData[i - 1].filename + "_to_" + AllData[j - 1].filename + ".xdelta";
				storage_graph[i][j] = getFileSize(deltaName, i - 1);
				AllData[j - 1].parent_index = i - 1;
			}
			if (storage_graph[i][j] == 0)
			{
				cout << " i = " << i << "  j = " << j << endl;
				if (i != 0)
					cout << AllData[i - 1].filename << endl;
				cout << AllData[j - 1].filename << " ?? " << endl;
				exit(0);
			}
			// recreation_graph[i][j] = storage_graph[i][j];
			// recreation_graph[i][j] = singleRecoverTime(j - 1);
			overhead += produceDeltaTime(j - 1);
		}
	}

	if (1)
	{
		// for test
		ofstream out;
		out.open("test.csv");
		for (int i = 0; i < n + 1; ++i)
		{
			for (int j = 0; j < n + 1; ++j)
				out << storage_graph[i][j] << ",";
			out << endl;
		}
		out.close();
	}

	// 计时器
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// MCA
	MCA mca(storage_graph);
	mca.calMCA();

	if (1)
		cout << "MAC finish." << endl;
}

vector<double> readMaxRecreationCost()
{
	// 存储读取的数据
	vector<double> data;
	ifstream fin;
	fin.open("maxRecreation.csv");

	if (!fin.good())
		return data;

	string buf;
	// 开始读取
	getline(fin, buf); // 第一行不读
	while (getline(fin, buf))
	{
		int num = atof(buf.c_str());
		data.emplace_back(num);
	}

	fin.close();
	return data;
}
