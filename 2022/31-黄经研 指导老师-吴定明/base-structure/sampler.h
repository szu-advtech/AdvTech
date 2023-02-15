//采样器
#include<iostream>
#include<vector>
#include<string>
#include<assert.h>
#include "sampleTable.h"
#ifndef setting 
#include "setting.h"
#define setting
#endif
using namespace std;

class sampler{
  public:
  sampleTable* st;
  int window_len;
  long long cur_time;
  long long last_landmark;
  long long landmark;
  int edge_estimate;
  double tri_sample_prob;
  int hashindex;

  //采样器三个参数为采样数量，窗口大小，hash函数index
  sampler(int size, int window, int hashfun){
    st = new sampleTable(size);
    window_len = window;
    cur_time = 0;
    landmark = 0;
    last_landmark = 0;
    hashindex = hashfun;
  }

  //处理边
  void process_edge(unsigned int s_id, unsigned int d_id, long long t){
    if (s_id < d_id){       //统一将边的起点设为大于终点
      unsigned int temp = s_id;
			s_id = d_id;
			d_id = temp;
    }
    string s_str = to_string(s_id);
    string d_str = to_string(d_id);
    string e_str = s_str + d_str;
    double p = double((*hashfun[hashindex])((const unsigned char*)e_str.c_str(), e_str.length()) % 1000000+1) / 1000001;   //获取优先级		
		cur_time = t;
    //更新sampleTable，处理边
    st->update_edge(t - window_len, landmark, last_landmark);
    if(t - landmark >= window_len){     //距离当前lnew超过一个窗口长度，更新lnew位置
      assert(t - landmark < 2 * window_len);
      last_landmark = landmark;
      landmark = landmark + window_len;
      st->transform();
    }
    st->insert_edge(s_id, d_id, p, t, landmark, last_landmark, hashindex);    //到达的所有边都插入，由sampletable决定是否采样
  }

  //计算相关参数
  void count_para(){
    int sample_num = st->sample_num;
    double ak = 0.7213 / (1 + (1.079 / sample_num));  //论文中计算基数n的参数ak
    int total_edge_num = (double(ak * sample_num * sample_num) / (st->valid_para));   //两个slice的基数，即|W(T,lold)|
    int valid_sample_num = st->valid_edge_num;        //有效采样边数量
    if(total_edge_num < 2.5 * sample_num)             //数据量较小时特殊处理
      total_edge_num = -log(1 - double(st->edge_num) / sample_num) * sample_num;
    edge_estimate = total_edge_num * (double(st->valid_edge_num) / st->edge_num);    //n=|W(T,lold)|*m/M，为窗口内边数估计
    tri_sample_prob = (double(valid_sample_num) / edge_estimate) * 
                      (double(valid_sample_num - 1) / (edge_estimate - 1)) * 
                      (double(valid_sample_num - 2) / (edge_estimate - 2));
  }

  //计算三角形
  int count_global_tri(){
    return (st->valid_tri_num) / tri_sample_prob;
  }

  int count_local_tri(unsigned int v_id){
    node_sample* node = st->nodes_table->get_pos(v_id);
    if(!node)
      return 0;
    else 
      return (node->localCount) / tri_sample_prob;
  }
};

