//
// Created by yuanj on 2022/11/6.
// 测量吞吐量

#include <iostream>
#include "db_impl.h"
#include <chrono>
#include "config.h"
#include <random>
#include <stdlib.h>
#include <chrono>
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>

using namespace std;
using namespace chrono;
typedef long long ll;
vector<ll> under_data;

int main(int argc, char** argv) {
  string PATH = string(argv[1]);
  int op = atoi(argv[2]);  // 0: 性能优先  1: 存储优先
  double req = atoi(argv[3]);  // 性能或存储限制
  double read_percentage = atof(argv[4]);
  double write_percentage = atof(argv[5]);

  cout << "[Stage 1]: 从外部文件读取数据..." << "\n";
  ifstream fp(PATH);
  string line;
  // getline(fp, line);
  while (getline(fp, line)) {
    istringstream readstr(line);
    string number;
    getline(readstr, number, ',');
    under_data.push_back(atoll(number.data()));
  }
  cout << "数据规模: " << under_data.size() << "\n";

  // 挑选e值
  // cout << "[Stage 2]: 使用Cost Model返回最佳e值..." << "\n";
  // config::ERROR = get_e(op, req, under_data);
  // assert(op == 0 || op == 1);
  // assert(config::ERROR == 10 || config::ERROR == 100 || config::ERROR == 1000);
  cout << "e值: " << config::ERROR << "\n";

  cout << "[Stage 3]: 建立FITing-tree..." << "\n";
  // vector<ll> first_in_data(under_data.begin(), under_data.begin() + int(under_data.size() * read_percentage));
  // construct(first_in_data);
  construct(under_data);

  default_random_engine e(255);
  uniform_int_distribution<uint64_t> uniform_dist_file(0, under_data.size() - 1);
  uniform_int_distribution<uint64_t> uniform_dist_file2(0, 1000000);
  double totle_time = 0;
	ll cnt = 0;
  int rwop = 0;  // 概率操作, < 为读操作, > 为写操作
  srand((unsigned)time(nullptr));
  cout << "[Stage 4]: 混合读写..." << "\n";
	while(true) {
    rwop = rand() % 10;
    // 读操作
    if (rwop < read_percentage * 10) {
      ll tk = uniform_dist_file(e);
      tk = under_data[tk];
      auto st = system_clock::now();
      get(tk);
      auto en = system_clock::now();
      auto duration = duration_cast<microseconds>(en - st);
      totle_time += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }
    // 写操作
    else {
      ll tk = uniform_dist_file2(e);
      auto st = system_clock::now();
      insert(tk);
      auto en = system_clock::now();
      auto duration = duration_cast<microseconds>(en - st);
      totle_time += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }
		if (totle_time > 1.0) {
				break;
		}
		cnt += 1;
	}
  cout << "混合吞吐量: " << cnt << "\n";
  return 0;
}
