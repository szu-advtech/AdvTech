//存储采样的结构（节点，边，相关计数）
#include <vector>
#include <assert.h>
#include<math.h>
#include<iostream>
#ifndef setting 
#include "setting.h"
#define setting
#endif
using namespace std;

class sampleTable{
  public:
  nodesTable* nodes_table;
  edgesTable* edges_table;
  int sample_num;       //采样数，substream个数k
  int node_num;         //采样节点数
  int edge_num;         //采样边数
  int valid_edge_num;   //有效采样边
  int valid_tri_num;    //有效三角形数
  double valid_para;    //有效参数（用于计算|W(T,lold)|），论文中的Σ(2^-R[i])
  int pre_edge_num;     //预处理边数
  int pre_tri_num;      //预处理三角形数
  double pre_para;      //预处理参数

  sampleTable(int size){
    nodes_table = new nodesTable(4, size*2);  //4为将表划分为4段
    edges_table = new edgesTable(size);
    sample_num = size;
    node_num = 0;
    edge_num = 0;
    valid_edge_num = 0;
    valid_tri_num = 0;
    valid_para = size;      //?
    pre_edge_num = 0;
    pre_tri_num = 0;
    pre_para = size;
  }

  //修改三角形，type=1时表示增加，=-1时表示减少
  //输入一条边，节点为s和d，遍历这两个节点相关的采样边，将其另一个节点存入容器中，检查是否有相同节点，即共同邻居，若有则得到一个三角形
  void set_triangle(node_sample* s, node_sample* d, long long last_landmark, int type){
    vector<unsigned int> v1;      //存储节点，用于计算共同邻居，三角形等属性
    vector<unsigned int> v2;
    unsigned int node_s = s->nodeId;
    unsigned int node_d = d->nodeId;
    int edge_s = s->firstEdgeId;
    int edge_d = d->firstEdgeId;
    while(edge_s >= 0){       //遍历所有与节点s连接的边
      unsigned int temp;
      int next_edge;
      if(edges_table->table[edge_s].s == node_s){       //在边表中找到对应边，并找到节点s
        temp = edges_table->table[edge_s].d;            //先将这条边的另外一个点保存
        next_edge = edges_table->table[edge_s].crossList[last_s];   //获取上一条边，注意是图结构的上一条边,也就是另一条和s连接的边
      }else if(edges_table->table[edge_s].d == node_s){
        temp = edges_table->table[edge_s].s;
        next_edge = edges_table->table[edge_s].crossList[last_d];
      }

      //只计算有效边
      //两种情况：1.cand为测试边且已经二次过期 2.cand已经从测试边更新为候选边，此时它在采样边之后
      if(edges_table->table[edge_s].cand.t < last_landmark || edges_table->table[edge_s].cand.t > edges_table->table[edge_s].t)
        v1.push_back(temp);     //加入的是这条边的另一个节点,为了找共同邻居
      edge_s = next_edge;
    }

    while(edge_d >= 0){
      unsigned int temp;
      int next_edge;
      if(edges_table->table[edge_d].d == node_d){
        temp = edges_table->table[edge_d].s;
        next_edge = edges_table->table[edge_d].crossList[last_d];
      }else if(edges_table->table[edge_d].s == node_d){
        temp = edges_table->table[edge_d].d;
        next_edge = edges_table->table[edge_d].crossList[last_s];
      }
      if(edges_table->table[edge_d].cand.t < last_landmark || edges_table->table[edge_d].cand.t > edges_table->table[edge_d].t)
        v2.push_back(temp);
      edge_d = next_edge;
    }
    vector<unsigned int> common_neighbor;
    count_common(v1, v2, common_neighbor);
    for(int i = 0; i < common_neighbor.size(); i++){
      int n = common_neighbor[i];
      nodes_table->get_pos(n)->localCount += type;
      s->localCount += type;
      d->localCount += type;
      valid_tri_num += type;
    }
    //删除相关结构
    common_neighbor.clear();
    v1.clear();
    v2.clear();
    vector<unsigned int>().swap(common_neighbor);
    vector<unsigned int>().swap(v1);
    vector<unsigned int>().swap(v2);
  }

  void set_pre_triangle(node_sample* s, node_sample* d, long long landmark, int type){
    vector<unsigned int> v1;      
    vector<unsigned int> v2;
    unsigned int node_s = s->nodeId;
    unsigned int node_d = d->nodeId;
    int edge_s = s->firstEdgeId;
    int edge_d = d->firstEdgeId;
    while(edge_s >= 0){       
      unsigned int temp;
      int next_edge;
      if(edges_table->table[edge_s].s == node_s){       
        temp = edges_table->table[edge_s].d;            
        next_edge = edges_table->table[edge_s].crossList[last_s];   
      }else if(edges_table->table[edge_s].d == node_s){
        temp = edges_table->table[edge_s].s;
        next_edge = edges_table->table[edge_s].crossList[last_d];
      }

      //只计算位于当前slice的vision counting
      if(edges_table->table[edge_s].t >= landmark)
        v1.push_back(temp);
      edge_s = next_edge;
    }

    while(edge_d >= 0){
      unsigned int temp;
      int next_edge;
      if(edges_table->table[edge_d].d == node_d){
        temp = edges_table->table[edge_d].s;
        next_edge = edges_table->table[edge_d].crossList[last_d];
      }else if(edges_table->table[edge_d].s == node_d){
        temp = edges_table->table[edge_d].d;
        next_edge = edges_table->table[edge_d].crossList[last_s];
      }
      if(edges_table->table[edge_d].t >= landmark)
        v2.push_back(temp);
      edge_d = next_edge;
    }
    vector<unsigned int> common_neighbor;
    count_common(v1, v2, common_neighbor);
    for(int i = 0; i < common_neighbor.size(); i++){
      int n = common_neighbor[i];
      nodes_table->get_pos(n)->visionCount += type;
      s->visionCount += type;
      d->visionCount += type;
      pre_tri_num += type;
    }
    common_neighbor.clear();
    v1.clear();
    v2.clear();
  }

  void set_both(node_sample* s, node_sample* d, long long last_landmark, long long landmark, int type){
    set_triangle(s, d, last_landmark, type);
    set_pre_triangle(s, d, landmark, type);
    return;
  }

  //修改十字链表，s和d为新插入的边的节点，s_id和d_id为对应节点id，pos为此边在边表中的位置
  void set_crossList(node_sample* s, node_sample* d, int pos){
    unsigned int s_id = s->nodeId;
    unsigned int d_id = d->nodeId;

    //头插法，先对当前边修改
    edges_table->table[pos].set_last_s(s->firstEdgeId);
    edges_table->table[pos].set_last_d(d->firstEdgeId);

    //再对被当前边修改影响到的边进行修改
    if(s->firstEdgeId >= 0){    
      if(edges_table->table[s->firstEdgeId].s == s_id)
        edges_table->table[s->firstEdgeId].set_next_s(pos);
      else
        edges_table->table[s->firstEdgeId].set_next_d(pos);
    }
    if(d->firstEdgeId >= 0){
      if(edges_table->table[d->firstEdgeId].s == d_id)
        edges_table->table[d->firstEdgeId].set_next_s(pos);
      else
        edges_table->table[d->firstEdgeId].set_next_d(pos);
    }

    s->set_firstEdgeId(pos);    //将当前边设为节点s最新的边，注意
    d->set_firstEdgeId(pos);
  }

  //删除一条边,修改十字链表，删除相关节点
  void delete_edge(node_sample* s, node_sample* d, int pos){
    unsigned int s_id = s->nodeId;
    unsigned int d_id = d->nodeId;
    int edge_last_s = edges_table->table[pos].crossList[last_s];
    int edge_last_d = edges_table->table[pos].crossList[last_d];
    int edge_next_s = edges_table->table[pos].crossList[next_s];
    int edge_next_d = edges_table->table[pos].crossList[next_d];

    if(s->firstEdgeId == pos){    //当前删除的边就是其节点的最新边
      if(edge_last_s < 0){        //没有在其前面的边，直接把节点删除
        s = NULL;
        nodes_table->delete_node(s_id);
        node_num--;
      }else{
        s->firstEdgeId = edge_last_s;   //用其前面的边来替代它
      }
    }

    if(d->firstEdgeId == pos){
      if(edge_last_d < 0){
        d = NULL;
        nodes_table->delete_node(d_id);
        node_num--;
      }else{
        d->firstEdgeId = edge_last_d;
      }
    }

    if(edge_last_s >= 0){       //当前边的前面有边，将其前面的边和其后面的边连在一起
      if(edges_table->table[edge_last_s].s == s_id)
        edges_table->table[edge_last_s].set_next_s(edge_next_s);
      else
        edges_table->table[edge_last_s].set_next_d(edge_next_s);
    }

    if(edge_next_s >= 0){
      if(edges_table->table[edge_next_s].s == s_id)
        edges_table->table[edge_next_s].set_last_s(edge_last_s);
      else
        edges_table->table[edge_next_s].set_last_d(edge_last_s);
    }

    if(edge_last_d >= 0){
      if(edges_table->table[edge_last_d].d == d_id)
        edges_table->table[edge_last_d].set_next_d(edge_next_d);
      else
        edges_table->table[edge_last_d].set_next_s(edge_next_d);
    }

    if(edge_next_d >= 0){
      if(edges_table->table[edge_next_d].d == d_id)
        edges_table->table[edge_next_d].set_last_d(edge_last_d);
      else
        edges_table->table[edge_next_d].set_last_s(edge_last_d);
    }
  }

  void insert_edge(unsigned int s_id, unsigned int d_id, double p, long long t, long long landmark, long long last_landmark, int hashIndex){
    string s_str = to_string(s_id);
    string d_str = to_string(d_id);
    string edge_str = s_str + d_str;
    unsigned int pos = (*hashfun[hashIndex+1])((unsigned char*)(edge_str.c_str()), edge_str.length()) % sample_num;

    if(edges_table->table[pos].cand.t < last_landmark && edges_table->table[pos].cand.t >= 0)     //测试边失效，删除
      edges_table->table[pos].cand.reset();

    //之前没有采样边(前面的slice没有采样边)
    if(edges_table->table[pos].s == 0 && edges_table->table[pos].d == 0){             
      pre_edge_num++;
      pre_para = pre_para - 1 + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
      node_sample* s = nodes_table->get_pos(s_id);      //节点位置
      node_sample* d = nodes_table->get_pos(d_id);
      if(!s){         //若新节点不在表中，插入
        s = nodes_table->insert_node(s_id);
        node_num++;
      }
      if(!d){
        d = nodes_table->insert_node(d_id);
        node_num++;
      }
      if(edges_table->table[pos].cand.t >= 0){       //没有采样边但是有test边
        assert(edges_table->table[pos].cand.t < landmark && edges_table->table[pos].cand.t >= last_landmark);   //test边在lold和lnew之间
        if(edges_table->table[pos].cand.p <= p){     //当前边优先级大于test边，直接替换
          valid_edge_num++;
          edges_table->set_sample(pos, s_id, d_id, p, t);     //替换采样边
          set_crossList(s, d, pos);      //修改十字链表
          set_both(s, d, last_landmark, landmark, 1);    //更新三角形（增加）
          valid_para = valid_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].cand.p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
					edges_table->table[pos].cand.reset();
        }else{                        //当前边优先级小于test边
          edges_table->set_sample(pos, s_id, d_id, p, t);
          set_crossList(s, d, pos);
          set_pre_triangle(s, d, landmark, 1);    //只更新预计数
        }
      }else{          //没有采样边和test边，将当前边设为采样边
        edge_num++;
        valid_edge_num++;
        edges_table->set_sample(pos, s_id, d_id, p, t);
        set_crossList(s, d, pos);
        set_both(s, d, landmark, last_landmark, 1);     //实际计数和预计数都更新
        valid_para = valid_para - 1 + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
      }
      return;
    }

    //当前插入的边已经是采样边
    if(edges_table->table[pos].s == s_id && edges_table->table[pos].d == d_id){
      if(edges_table->table[pos].t < landmark){    //当前采样边在lnew之前（仍在窗口内），累加预计数
        pre_para = pre_para - 1 + 1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1);
        pre_edge_num++;
        node_sample* s = nodes_table->get_pos(s_id);
        node_sample* d = nodes_table->get_pos(d_id);
        set_pre_triangle(s, d, landmark, 1);
      }
      edges_table->update_sample(pos, t);     //更新采样边的时间戳
      if(edges_table->table[pos].cand.p < edges_table->table[pos].p && edges_table->table[pos].cand.t <= edges_table->table[pos].t)     //测试边的优先级较小且时间较小，可以删除
        edges_table->delete_cand(pos);
      return;
    }

    //之前的采样边在lnew之前，滑动窗口内
    if(edges_table->table[pos].t < landmark){
      if(edges_table->table[pos].p <= p){        //当前边的优先级大于采样边的优先级，直接替换
        assert(edges_table->table[pos].cand.t >= landmark || edges_table->table[pos].cand.t < 0);   //采样边的候选边要么在landmark之后要么没有
        edges_table->delete_cand(pos);
        node_sample* old_s = nodes_table->get_pos(edges_table->table[pos].s);
        node_sample* old_d = nodes_table->get_pos(edges_table->table[pos].d);
        valid_para = valid_para + -1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);    //减去旧的参数加上新的参数
        pre_para = pre_para -1 + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
        pre_edge_num++;
        set_triangle(old_s, old_d, last_landmark, -1);
        delete_edge(old_s, old_d, pos);
        edges_table->set_sample(pos, s_id, d_id, p ,t);

        node_sample* s = nodes_table->get_pos(s_id);
        node_sample* d = nodes_table->get_pos(d_id);
        if(!s){
          s = nodes_table->insert_node(s_id);
          node_num++;
        }
        if(!d){
          d = nodes_table->insert_node(d_id);
          node_num++;
        }
        set_crossList(s, d, pos);
        set_both(s, d, last_landmark, landmark, 1);
      }else{        //优先级小于采样边的优先级，检查候选边
        if(edges_table->table[pos].cand.p <= p){
          edges_table->set_cand(pos, s_id, d_id, p, t);
        }
      }
    }else{        //采样边在lnew之后
      if(edges_table->table[pos].p <= p){
        node_sample* old_s = nodes_table->get_pos(edges_table->table[pos].s);
        node_sample* old_d = nodes_table->get_pos(edges_table->table[pos].d);
        if(edges_table->table[pos].cand.t < landmark && edges_table->table[pos].cand.t >= 0){   //test边
          assert(edges_table->table[pos].cand.t >= last_landmark);                 //test边仍然有效
          if(edges_table->table[pos].cand.p <= p){             //当前边的优先级大于test
            valid_para = valid_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].cand.p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
            pre_para = pre_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
            edges_table->delete_cand(pos);
            set_pre_triangle(old_s, old_d, landmark, -1);       //oldsample并不是真实采样边
            delete_edge(old_s, old_d, pos);
            edges_table->set_sample(pos, s_id, d_id, p, t);
            node_sample* s = nodes_table->get_pos(s_id);
            node_sample* d = nodes_table->get_pos(d_id);
            if(!s){
              s = nodes_table->insert_node(s_id);
              node_num++;
            }
            if(!d){
              d = nodes_table->insert_node(d_id);
              node_num++;
            }
            set_crossList(s, d, pos);
            set_both(s, d, last_landmark, landmark, 1);
            valid_edge_num++;
          }else{            //当前边优先级小于test
            pre_para = pre_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
            set_pre_triangle(old_s, old_d, landmark, -1);
            delete_edge(old_s, old_d, pos);
            edges_table->set_sample(pos, s_id, d_id, p, t);
            node_sample* s = nodes_table->get_pos(s_id);
            node_sample* d = nodes_table->get_pos(d_id);
            if(!s){
              s = nodes_table->insert_node(s_id);
              node_num++;
            }
            if(!d){
              d = nodes_table->insert_node(d_id);
              node_num++;
            }
            set_crossList(s, d, pos);
            set_pre_triangle(s, d, landmark, 1);    //新边此时不是有效采样边
          }
        }else{        //没有候选边或test边，直接替换采样边
          assert(edges_table->table[pos].cand.t >= 0);
          valid_para = valid_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
          pre_para = pre_para - 1/pow(2, int(-(log(1 - edges_table->table[pos].p) / log(2)))+1) + 1/pow(2, int(-(log(1 - p) / log(2)))+1);
          set_both(old_s, old_d, last_landmark, landmark, -1);
          delete_edge(old_s, old_d, pos);
          edges_table->set_sample(pos, s_id, d_id, p, t);
          node_sample* s = nodes_table->get_pos(s_id);
          node_sample* d = nodes_table->get_pos(d_id);
          if(!s){
            s = nodes_table->insert_node(s_id);
            node_num++;
          }
          if(!d){
            d = nodes_table->insert_node(d_id);
            node_num++;
          }
          set_crossList(s, d, pos);
          set_both(s, d, last_landmark, landmark, 1);
        }
      }
    }
    return;
  }

  //更新边，t为更新后窗口尾部的时间，landmark为lnew，last_landmark为lold
  void update_edge(long long t, long long landmark, long long last_landmark){
    int timeList_pos = edges_table->timeList_head;     //最早的一条边的位置
    if(timeList_pos < 0)
      return;
    int pos = timeList_pos % sample_num;              //对应在table中的位置
    while(edges_table->table[pos].t < t){     //边失效
      timeList_pos = edges_table->table[pos].t_next;     //获取下一条边的位置
      if(edges_table->table[pos].cand.t < last_landmark && edges_table->table[pos].cand.t >= 0)   //若test边已失效，删除
        edges_table->table[pos].cand.reset();
      if(edges_table->table[pos].cand.t >= t){           //若test边未失效
        pre_para = pre_para - 1 + 1/pow(2, int(-(log(1 - edges_table->table[pos].cand.p) / log(2)))+1);
        node_sample* old_s = nodes_table->get_pos(edges_table->table[pos].s);
        node_sample* old_d = nodes_table->get_pos(edges_table->table[pos].d);
        set_triangle(old_s, old_d, last_landmark, -1);
        delete_edge(old_s, old_d, pos);
        edges_table->delete_sample(pos);    //删除过期边
        edge_sample temp = edges_table->table[pos];
        valid_edge_num--;
        //将候选边设为采样边
        edges_table->table[pos].reset(temp.cand.s, temp.cand.d, temp.cand.p, temp.cand.t, temp.cand.t_prev, temp.cand.t_next);
        //修改时间序列表
        edges_table->set_pointer(temp.cand.t_prev, 1, pos);
        edges_table->set_pointer(temp.cand.t_next, 0, pos);
        //修改位置，原本候选边是pos+size，现在改为pos
        if(edges_table->timeList_tail == pos + sample_num)
          edges_table->timeList_tail = pos;
        if(edges_table->timeList_head == pos + sample_num)
          edges_table->timeList_head = pos;
        //将节点加入节点表
        node_sample* s = nodes_table->get_pos(temp.cand.s);
        node_sample* d = nodes_table->get_pos(temp.cand.d);
        if(!s){
          s = nodes_table->insert_node(temp.cand.s);
          node_num++;
        }
        if(!d){
          d = nodes_table->insert_node(temp.cand.d);
          node_num++;
        }
        set_crossList(s, d, pos);
        set_pre_triangle(s, d, landmark, 1);      //插入的还不是有效边
        pre_edge_num++;
        //将旧的采样边变为测试边
        edges_table->table[pos].cand.reset(temp.s, temp.d, temp.p, temp.t);
      }else{      //没有测试边，直接更新三角形，删除采样边
        node_sample* old_s = nodes_table->get_pos(edges_table->table[pos].s);
				node_sample* old_d = nodes_table->get_pos(edges_table->table[pos].d);
        set_triangle(old_s, old_d, last_landmark, -1);
        delete_edge(old_s, old_d, pos);
        valid_edge_num--;
        edges_table->table[pos].cand.reset(edges_table->table[pos].s, edges_table->table[pos].d, edges_table->table[pos].p, edges_table->table[pos].t);
        edges_table->delete_sample(pos);
      }
      if(timeList_pos < 0)
        break;
      pos = timeList_pos % sample_num;    //检查下一条边
    }
  }

  void transform(){             //对实际计数和预计数进行转化
    valid_tri_num = pre_tri_num;
    valid_edge_num = pre_edge_num;
    valid_para = pre_para;
    edge_num = pre_edge_num;
    pre_tri_num = 0;
    pre_edge_num = 0;
    pre_para = sample_num;
    nodes_table->transform();
  }      
};