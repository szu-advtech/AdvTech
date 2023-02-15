//节点表，存储采样到的节点
#include <string>
using namespace std;

class nodesTable{
  public:
  node_sample* table;   //table是存储采样节点的数组（每个位置还有链表），且切分成了n段，节点使用n个hash函数映射，最后选择一个最优的位置进行存储，为了均匀映射且形成的链表最短
  int len;    //节点表的长度，即采样节点的数量，即采样边数（substream数）*2
  int n;      //hash函数数量
  int unit_len; //table被划分成n段，每段长度为unit_len

  nodesTable(int n_num, int l_num){
    n = n_num;
    len = l_num;
    unit_len = l_num/n_num;
    table = new node_sample[len];
    for(int i = 0; i < len; i++){
      table[i].init(0);
    }
  }

  //此处的析构函数先不写

  node_sample* insert_node(unsigned int node, int edge = -1);
  node_sample* set_firstEdge(unsigned int node, int edge);
  int get_firstEdge(unsigned int node);
  node_sample* get_pos(unsigned int node);
  void delete_node(unsigned int node);
  void transform();   //将vision counting转化为真实采样
};

node_sample* nodesTable::insert_node(unsigned int node, int edge){
  int min_len = 0x7fffffff;     //节点连接的链表的最小长度
  int min_pos = -1;             //min_len对应位置下标
  int empty_pos = -1;           //记录找到的空位置下标
  string node_str = to_string(node);

  //先找位置
  for(int i = 0; i < n; i++){     //使用n个hash函数对node进行映射
    unsigned int pos = (*hashfun[i])((unsigned char*)(node_str.c_str()), node_str.length()) % unit_len;  //hash函数映射到一个相对位置
    int address = unit_len * i + pos;      //计算在table中的位置

    if(table[address].nodeId == 0){               //情况1：当前位置为空
      if(empty_pos < 0)
        empty_pos = address;
    }else if(table[address].nodeId == node){      //情况2：当前位置存储的就是当前节点
      if(edge>0)
        table[address].set_firstEdgeId(edge); //更新节点的连接边
      return &(table[address]);
    }else{                                        //情况3：当前位置存储的是其他节点，查找其链表中是否有当前节点
      int count = 0;    //链表长度
      node_sample* temp = table[address].next;
      while(temp){
        count++;
        if(temp->nodeId == node){     //找到当前节点
          if(edge > 0)
            temp->set_firstEdgeId(edge);
          return temp;
        }
        temp = temp->next;
      }
      if(count < min_len){      //更新最短链表长度
        min_len = count;
        min_pos = address;
      }
    }
  }

  //实际插入操作
  //有空位置直接插入
  if(empty_pos >= 0){
    table[empty_pos].nodeId = node;
    table[empty_pos].set_firstEdgeId(edge);
    return &(table[empty_pos]);
  }
  //没有空位置就将其插入到最短链表的末尾
  node_sample* temp = table[min_pos].next;
  if(!temp){
    temp = new node_sample(node, edge);
    table[min_pos].next = temp;
  }else{
    node_sample* last = temp;
    while(temp){
      last = temp;
      temp = temp->next;
    }
    temp = new node_sample(node, edge);
    last->next = temp;
  }
  return temp;
}

//找到节点位置并修改边
node_sample* nodesTable::set_firstEdge(unsigned int node, int edge){
  string node_str = to_string(node);
  for(int i = 0; i < n; i++){
    unsigned int pos = (*hashfun[i])((unsigned char*)(node_str.c_str()), node_str.length()) % unit_len;
    int address = i * unit_len + pos;

    if(table[address].nodeId == node){
      table[address].set_firstEdgeId(edge);
      return &(table[address]);
    }else{
      node_sample* temp = table[address].next;
      while(temp){
        if(temp->nodeId == node){
          temp->set_firstEdgeId(edge);
          return temp;
        }
        temp = temp->next;
      }
    }
  }
  return NULL;
}

//找到节点位置并返回边
int nodesTable::get_firstEdge(unsigned int node){
  string node_str = to_string(node);
  for(int i = 0; i < n; i++){
    unsigned int pos = (*hashfun[i])((unsigned char*)(node_str.c_str()), node_str.length()) % unit_len;
    int address = i * unit_len + pos;

    if(table[address].nodeId == node)
      return table[address].firstEdgeId;
    else{
      node_sample* temp = table[address].next;
      while(temp){
        if(temp->nodeId == node)
          return temp->firstEdgeId;
        temp = temp->next;
      }
    }
  }
  return -1;
}


node_sample * nodesTable::get_pos(unsigned int node){
  string node_str = to_string(node);
  for(int i = 0; i < n; i++){
    unsigned int pos = (*hashfun[i])((unsigned char*)(node_str.c_str()), node_str.length()) % unit_len;
    int address = i * unit_len + pos;

    if(table[address].nodeId == node)
      return &(table[address]);
    else{
      node_sample* temp = table[address].next;
      while(temp){
        if(temp->nodeId == node)
          return temp;
        temp = temp->next;
      }
    }
  }
  return NULL;
}

void nodesTable::delete_node(unsigned int node){
  string node_str = to_string(node);
  for(int i = 0; i < n; i++){
    unsigned int pos = (*hashfun[i])((unsigned char*)(node_str.c_str()), node_str.length()) % unit_len;
    int address = i * unit_len + pos;     
    
    if(table[address].nodeId == node){    //删除的节点在table上
      if(table[address].next){            //用其链表上后一个节点进行替换
        node_sample* cur = table[address].next;
        table[address].firstEdgeId = cur->firstEdgeId;
        table[address].next = cur->next;
        table[address].nodeId = cur->nodeId;
        delete cur;
      }else
        table[address].reset();          //没有后续节点时直接把当前节点删除
      return;
    }else{                    //删除的节点在链表上
      node_sample* temp = table[address].next;
      node_sample* last = temp;       //temp的上一个节点
      while(temp){
        if(temp->nodeId == node){     //找到当前节点
          if(last == temp){           //last指向最后一个节点了
            table[address].next = temp->next;
            delete temp;
          }else{
            last->next = temp->next;  //调整指针
            delete temp;
          }
          return;
        }
        last = temp;
        temp = temp->next;
      }
    }
  }
  return;
}

//将vision counting转化为实际计数(仅转化节点的本地计数)
void nodesTable::transform(){
  for (int i = 0; i < len; i++)
	{
		table[i].localCount = table[i].visionCount;		
		table[i].visionCount = 0;
		node_sample* temp = table[i].next;
		while (temp)
		{
			temp->localCount = temp->visionCount;
			temp->visionCount = 0;
			temp = temp->next;
		}
	}
}