//定义边的数据结构

#include<sstream>
#define next_s 0
#define next_d 1
#define last_s 2
#define last_d 3
using namespace std;

//候选边
//在BPS中就是介于一次和两次过期之间的测试边
//在SWTC中，是前一个slice过期的最大边（此时作为测试边）或后一个slice未被选为采样边的最大边
class edge_cand{
  public:
  unsigned int s,d;
  double p;
  long long t;
  int t_prev;
  int t_next;
  //注意候选边不加入采样图中，故没有交叉链表的属性

  edge_cand(){
    s = 0;
    d = 0;
    p = -1;
    t = -1;
    t_prev = -1;
    t_next = 1;
  }

  void reset(unsigned int snum = 0, unsigned int dnum = 0, double priority = -1, long long time = -1, int prev = -1, int next = -1){
    s = snum;
    d = dnum;
    p = priority;
    t = time;
    t_prev = prev;
    t_next = next;
  }
};

class edge_sample{
  public:
  unsigned int s, d;      //边的起点和终点
  double p;
  long long t;
  int crossList[4];    //十字链表，一条边保存四个指针，指向它两个节点的其他边，主要表示边之间的连接关系，节点通过firstedge与一条边连接，并通过这条边找到其他的边
  int t_prev;          //指向时间序列上的前一条边，若样本表长度为m，则0~m-1表示采样边，m~2m-1指候选边，在逻辑上整个序列连成一个时间表
  int t_next;
  edge_cand cand;      //候选边

  edge_sample(){
    s = 0;
    d = 0;
    p = -1;
    t = -1;
    for(int i = 0; i < 4; i++) crossList[i] = -1;
    t_prev = -1;
    t_next = -1;
    cand.reset();
  }

  void set_next_s(int s){
    crossList[next_s] = s;
  }
  void set_next_d(int d){
    crossList[next_d] = d;
  }
  void set_last_s(int s){
    crossList[last_s] = s;
  }
  void set_last_d(int d){
    crossList[last_d] = d;
  }
  void reset(unsigned int snum = 0, unsigned int dnum = 0, double priority = -1, long long time = -1, int prev = -1, int next = -1){
    s = snum;
    d = dnum;
    p = priority;
    t = time;
    t_prev = prev;
    t_next = next;
    for(int i = 0; i < 4; i++){
      crossList[i] = -1;
    }
  }    
};

