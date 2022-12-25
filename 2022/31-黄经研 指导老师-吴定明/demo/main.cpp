#include<fstream>
#include<string>
#include<iostream>
#include<vector>
#include<time.h>
#include "../base-structure/sampler.h"
using namespace std;

int main(){
  unsigned int s_id, d_id;
  long long t;
  double time_unit = 20;      // 窗口长度的单位，时间戳的极差/边的数量
  int window_len = 4000000;
  int hashindex = 0;
  double sample_rate = 0.04;     //采样率
  long wsize = window_len * time_unit;   //窗口长度（时间戳）
  int sample_num = window_len * sample_rate;  //采样数
  long cur_time = 0;
  string file_in = "D:\\VSCode\\workplace\\SWTC\\data\\actor_d0.txt";
  string file_out = "D:\\VSCode\\workplace\\SWTC\\data\\result.txt";
  ifstream fin(file_in.c_str());
  ofstream fout(file_out.c_str());

  sampler* sc = new sampler(sample_num, wsize, hashindex);    //swtc采样器
  long long t0 = -1;            //记录是否为第一个节点(此时将开始时间设为当前节点的时间戳)
  long long cur_point = 0;      //记录当前在第几个检查点，tmp_point = count / checkpoint
  int checkpoint = wsize / 10;  //每隔窗口长度的1/10设置一个检查点
  int edge_num = 0;
  while(fin>>s_id>>d_id>>t){
    if(t0 < 0)
      t0 = t;             //把进入的第一条边的时间戳设为开始时间戳
    s_id++;
    d_id++;
    if(s_id != d_id)
      cur_time = t - t0;  //cur_time记录当前时间与开始时间的距离，即相对时间
    else
      continue;
    edge_num++;
    sc->process_edge(s_id, d_id, cur_time);
    //时间小于窗口两倍时不设置检查点，因为此时没有足够的过期边，后面一个条件表示应该增加一个新的检查点了
    if(cur_time >= 2 * wsize && int(cur_time / checkpoint) > cur_point){
      srand((int)time(0));    //生成随机数
      cur_point = cur_time / checkpoint;
      sc->count_para();
      fout<<sc->st->valid_edge_num<<' '<<sc->edge_estimate<<' '<<sc->st->valid_tri_num<<' '<<sc->count_global_tri()<<endl;
      fout<<endl;
      cout<<sample_num<<" check point " << cur_point << endl;
    }
  }
  fin.close();
  fout.close();
  delete sc;
  return 0;
}