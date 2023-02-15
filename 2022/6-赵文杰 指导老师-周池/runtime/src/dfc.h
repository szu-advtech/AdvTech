#include<threadpool.h>

#ifndef _DFC_H_
#define _DFC_H_

#ifdef __cplusplus
extern "C" {
#endif

#define INTSIZE 4
struct DF_Ready_Flags;
struct DF_Fun_DC;
struct Active_Data;
struct DF_TargetADList;
struct DF_Fun_Node;
struct DF_TargetFunList;
struct DF_SourceItem;
typedef struct DF_Ready_Flags DF_Ready;
typedef struct DF_Fun_DC DF_DC;
typedef struct Active_Data DF_AD;
typedef struct DF_TargetADList DF_TADL;
typedef struct DF_Fun_Node DF_FN;
typedef struct DF_TargetFunList DF_TFL;
typedef struct DF_SourceItem DF_SI;

struct DF_TargetFunList{
    int Num;         //输出目标的数据流节点数
    DF_FN **Target;      //输出目标的数据流函数指针线性表
    threadpool_t* pool;
    DF_SI* source_list;
    int source_list_len;
    DF_AD** output_list;
    int output_list_len;
    void** result;
    void (**Func_Target) (void *);
    int* item_index_order;
    int should_hash;
    int thread_num;
    int* thread_task;
};   //线性表结构

struct Active_Data{
    int Data_Count;                 //用于记录链上有多>少个数据对象可用    00 缓冲区数量   单位/个
    int Sum;                    //至今有过多少数据量           单位/个
    void *Data;
    int   Head;               //头尾 单位/个
    int   Tail;
    int   Order;
    // int * ?
    int   MaxSize;
    int   persize;//数据单位大小             单位字节       size+4
    int FanOut;                      //指向几个目标函数

    // 这里耶应该用函数指针列表吧
    DF_FN **DF_Fun_Index;           //指向目标数据流函>数的 数据块二级指针      指向指针的指针
    // 指向数据流函数的数据，目前来看 Active Data 应该>是唯一的
    int*   DF_flag_Index;          //是第几位
    pthread_rwlock_t lock;
};//主动数据链路

struct DF_Fun_Node {
    //DF_FN     * next_fun;                     //构成>线性链表
    //DF_FN     * next_ready_on_CPU;                //>挂入某个CPU的就绪队列
    //DF_sched_data sched_data;                 //调度>数据
    //DF_DAG_node   DAG_node;                       //>数据流图节点
    int item_index;                                //数据流节点编号
    void        (*Func)(void);                      //>数据流函数
    unsigned        char * Name;                    //>函数名
    DF_Ready*       ready;
    pthread_mutex_t ready_lock;                 //就绪队列锁---保证不同的FN输出为1个FN的输入就绪队列的访问
    int             FinishNum;                        //当前已经派发出去的任务数
    int             realFinishNum;                     //当前真正完成了几个工作，初始为0   用以从数据端获得正确数据
    int stop;  //停止运行标志
    pthread_rwlock_t finish_lock;               //fn 顺序锁    保证同一个fn的不同次执行的顺序   保证getdata
    pthread_rwlock_t realfinish_lock;           //保证updata
    DF_TADL*      DF_Fun_AD_InPut;                          //输入主动数据块
    // 每次运行时都要从这取数据吗？
    // 数据分配是怎样一个操作呢
    DF_TADL*      DF_Fun_AD_OutPut;                      //输出主动链块
    // 塞数据到里面
};

void DF_ADInit(DF_AD* AD, int persize,int FanOut);
void DF_FNInit1(DF_FN* FN,void*FunAddress ,char *Name, int InPutADNum,...);
void DF_FNInit2(DF_FN *FN, int OutPutADNum, ...);
void DF_AD_UpData(int DF_Count,DF_TFL *table, DF_FN *F,...);
int DF_AD_GetData(DF_FN* F, ...);
void DF_SOURCE_Get_And_Update(DF_FN* F, int* count);
void DF_Init(DF_TFL* table, int InputFNNum, ...);
void DF_Thread_Init(DF_TFL* table, int threadnum, int queuesize);
void DF_SourceInit(DF_TFL *table, int sourcenum, ...);
void DF_OutputInit(DF_TFL *table, int outputnum, ...);
void** DF_Result(DF_TFL *table);
void DF_Run (DF_TFL *table, int DF_Count);
void DF_Source_Stop(DF_TFL *table, int item_index);
void printf_thread_info(DF_TFL* table);
int use_funcname_to_get_item_index(DF_TFL *table,char* funcname);
#ifdef __cplusplus
}
#endif
#endif
