#include <string>
using namespace std;

class edgesTable{
  public:
  int size;     //边表的长度，即采样边数量
  int timeList_head;   //节点之间通过指针在逻辑上连成一个时间序列，此属性标志序列头，即最旧的边的id
  int timeList_tail;
  edge_sample* table;   //存储采样边的列表

  edgesTable(int num){    //num是采样边数量
    size = num;
    timeList_head = -1;
    timeList_tail = -1;
    table = new edge_sample[num];
    for(int i = 0; i < num; i++){
      table[i].reset();
      table[i].cand.reset();
    }
  }

  //修改指针（某个边的t_prev或t_next），index边在表中的位置，pointer为修改的指针（0为prev，1为next），aim为目标边的id
  void set_pointer(int index, int pointer, int aim){
    if(index == -1)
      return;
    if(index < size){     //小于采样边的长度为采样边
      if(pointer == 0)
        table[index].t_prev = aim;
      else
        table[index].t_next = aim;
    }else{
      if(pointer == 0)
        table[index % size].cand.t_prev = aim;    //采样边和候选边通过在表中的位置来区分（0~m-1或m~2m-1），但实际在table中还是存储在一起
      else
        table[index % size].cand.t_next = aim;
    }
  }

  //修改时间列表（修改前后指针+头尾指针）,type为0时表示修改的是采样边，为1表示修改候选边
  void set_timeList(int index, int type){
    
  }

  //修改当前位置的采样边
  void set_sample(int pos, unsigned int s, unsigned int d, double p, long long t){
    //由于修改十字链表需要节点表，故由上层结构（sampleTable）进行修改

    int index = pos;
    int prev = table[pos].t_prev;
    int next = table[pos].t_next;
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    //插入新的边，接在时间序列的尾部，故timeList_prev设为之前的tail，其他参数默认
    table[pos].reset(s, d, p, t, timeList_tail, -1);

    set_pointer(timeList_tail, 1, index);    //修改前一条边的指针
    timeList_tail = index;
    if(timeList_head == -1)          //当前时间列表为空
      timeList_head = index;
  }

  //修改当前位置的候选边
  void set_cand(int pos, unsigned int s, unsigned int d, double p, long long t){
    int index = pos + size;
    int prev = table[pos].cand.t_prev;
    int next = table[pos].cand.t_next;
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    table[pos].cand.reset(s, d, p, t, timeList_tail, -1);
    set_pointer(timeList_tail, 1, index);
    timeList_tail = index;
    if(timeList_head == -1)
      timeList_head = index;
  }

  void delete_sample(int pos){
    int index = pos;
    int prev = table[pos].t_prev;
    int next = table[pos].t_next;
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    table[pos].reset();     //删除采样边
  }

  void delete_cand(int pos){
    int index = pos + size;
    int prev = table[pos].cand.t_prev;
    int next = table[pos].cand.t_next;
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    
    table[pos].cand.reset();  //删除候选边
  }

  //更新当前位置边的时间戳
  void update_sample(int pos, long long time){
    int index = pos;
    int prev = table[pos].t_prev;
    int next = table[pos].t_next;
    table[pos].t = time;    //更新时间
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    
    table[pos].t_prev = timeList_tail;    //将边调整为时间列表上的最后一条
    table[pos].t_next = -1;
    set_pointer(timeList_tail, 1, index);   //调整前面的边的后续指针
    timeList_tail = index;
    if(timeList_head == -1)
      timeList_head = index;
  }

  void update_cand(int pos, long long time){
    int index = pos + size;
    int prev = table[pos].cand.t_prev;
    int next = table[pos].cand.t_next;
    table[pos].cand.t = time;    //更新时间
    set_pointer(prev, 1, next);
    set_pointer(next, 0, prev);
    if(timeList_head == index)
      timeList_head = next;
    if(timeList_tail == index)
      timeList_tail = prev;
    table[pos].cand.t_prev = timeList_tail;
    table[pos].cand.t_next = -1;
    set_pointer(timeList_tail, 1, index);
    timeList_tail = index;
    if(timeList_head == -1)
      timeList_head = index;
  }
};