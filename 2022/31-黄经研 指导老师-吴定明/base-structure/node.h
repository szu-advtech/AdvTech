//定义节点的数据结构
#include<sstream>
using namespace std;

//采样节点
class node_sample{
  public:
  unsigned int nodeId;
  int firstEdgeId;           //十字交叉链表的中该节点的第一条边，使用头插法，故是最新的边
  unsigned int localCount;    //本地三角形数
  unsigned int visionCount;   //预计数
  node_sample* next;

  //空参构造器
  node_sample(){
    nodeId = 0;
    firstEdgeId = -1;
    localCount = 0;
    visionCount = 0;
    next = NULL;
  }

  //构造器
  node_sample(unsigned int s, int edge = -1){
    nodeId = s;
    firstEdgeId = edge;
    localCount = 0;
    visionCount = 0;
    next = NULL;
  }

  void init(unsigned int s, int edge = -1)
	 {
	 	nodeId = s;
		firstEdgeId = edge;
		localCount = 0;
		visionCount = 0;
    next = NULL;
	 }

	 void set_firstEdgeId(int s)    //注意这里的s是在边表中的位置
	 {
	 	firstEdgeId = s;
	 }

	 void reset()
	 {
	 	nodeId = 0;
    firstEdgeId = -1;
    localCount = 0;
    visionCount = 0;
	 }
};