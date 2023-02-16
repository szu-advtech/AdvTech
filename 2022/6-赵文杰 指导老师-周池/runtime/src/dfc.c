#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#define THREADNUM  32
//#include <sys/sysinfo.h>

#include "threadpool.h"
#include "dfc.h"
#include "c_printf.h"

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

struct DF_Ready_Flags {
  int Flags;
  struct DF_Ready_Flags *next;
};

struct DF_TargetADList{
  int TargetNum;           //输入/输出主动数据数目
  DF_AD * Target[];   // 输入/输出主动数据指针 线性表
};

struct DF_SourceItem{
  int stop;
  int count;
  DF_FN* F;
};

void DF_ADInit(DF_AD* AD, int persize,int FanOut) {
  AD->Data_Count = 0; // 计数器为 0
  AD->Sum = 0;        // 总数据量为 0
  AD->MaxSize = 2; // 满队元素
  AD->Data = (void*)malloc(AD->MaxSize * (persize + INTSIZE)); // 初始化空间大小 15 个元素，persize 为数据单位大小(不记录数据调用次数，单纯的简单数据结构或者结构体)，而 INTSIZE 为其记录数（比如可调用次数）大小
  AD->Head = 0; // 好像是用循环队列存的，所以保存头尾，每次出头
  AD->Tail = 0;
  AD->Order = 0;
  AD->persize = persize + INTSIZE; // persize 为最终>大小，包括数据单位大小和一个计数器
  AD->FanOut = 0;//？ 这里应该是函数数目计数 为了使AD连向FN时正确所以我们先设置为0尽管已经知道了FANOUT的数量
  pthread_rwlock_init(&AD->lock, NULL);
  AD->DF_Fun_Index = (DF_FN**)malloc(sizeof(DF_FN*) * FanOut);   // 这个就是直接指向作为源数据的函数
  AD->DF_flag_Index = (int*)malloc(sizeof(int) * FanOut);   // 第几位这个用途待定
}

//FN>初始化函数一，  连接输入和FN节点的关系（FN指向AD，AD指向FN）
void DF_FNInit1(DF_FN* FN,void*FunAddress ,char *Name, int InPutADNum,...){
  int Flaglen;
  va_list ap;
  va_start(ap,InPutADNum);
  FN->DF_Fun_AD_InPut = (DF_TADL*)malloc(sizeof(DF_TADL) + InPutADNum * sizeof(DF_AD*));  //柔性数组使用的指针数组 可能有BUG
  FN->DF_Fun_AD_InPut->TargetNum = InPutADNum;
  Flaglen = (InPutADNum + 256 - 1) / 256; //InPutADNum/256 向上取整;
  for(int i = 0;i < InPutADNum;i++){
    DF_AD* p;
    p = va_arg(ap, DF_AD*);
    FN->DF_Fun_AD_InPut->Target[i] = p;     //Input指向AD
    p->DF_Fun_Index[p->FanOut] = (DF_FN*)FN;       //AD指向FUN
    p->DF_flag_Index[p->FanOut] = i;      //AD是FN>的第几位标志
    p->FanOut++;
  }
  FN->Func = (void (*)(void))FunAddress;   //初始化FN的一些常量信息，函数名，函数指针，就绪标志
  FN->Name = (unsigned char *)Name;
  FN->FinishNum = 0;
  FN->realFinishNum=0;
  FN->ready = 0;       
  pthread_rwlock_init(&FN->finish_lock, NULL);
  pthread_mutex_init(&FN->ready_lock, NULL);
  va_end(ap);
}

//FN初始化函数二，连接FN节点和输出AD关系
void DF_FNInit2(DF_FN *FN, int OutPutADNum, ...){
  va_list ap;
  va_start(ap,OutPutADNum);
  FN->DF_Fun_AD_OutPut = (DF_TADL*)malloc(sizeof(DF_TADL) + OutPutADNum * sizeof(DF_AD*));  //柔性数组使用的指针数组 可能有BUG
  FN->DF_Fun_AD_OutPut->TargetNum = OutPutADNum;
  for(int i = 0;i < OutPutADNum;i++){
    FN->DF_Fun_AD_OutPut->Target[i] = va_arg(ap,DF_AD*);
  }
  va_end(ap);
}

void DF_AD_UpData(int DF_Count,DF_TFL *table, DF_FN *F,...){       //地址，地址，地址
  va_list ap;
  va_start(ap,F);
  void *new_data_addr;
  int new_data_size;
  DF_AD *b;
  void *c;
  int Num=(F->DF_Fun_AD_OutPut->TargetNum);//一个函数输出的AD数量
  while(1){   //保证同一函数的先后次序
    if(DF_Count==F->realFinishNum+1)
    {
      pthread_rwlock_wrlock(&F->realfinish_lock);
      F->realFinishNum++;
      //    printf("%d 号任务正确  完成数%d\n",F->item_index,F->realFinishNum);
      break;
    }
    else
      usleep(3);
    
  }
  //如果没停止运行就允许更新数据
  if(F->stop==0)
  {
    // printf("%s writes %d\n", F->Name, DF_Count);
    for(int i=0;i<Num;i++  ){
      new_data_addr = va_arg(ap, void*);//新数据地址
      new_data_size = va_arg(ap, int);
      b = F->DF_Fun_AD_OutPut->Target[i];   //指向要跟新的AD
      pthread_rwlock_wrlock(&b->lock);
      if(b->MaxSize == b->Data_Count) {   //扩容
        // printf("%s扩容\n",F->Name );
        void *p=(void*)malloc(b->MaxSize*b->persize*2);

        memcpy(p, b->Data+b->Head*b->persize, (b->MaxSize - b->Head)*b->persize);
        memcpy(p+(b->MaxSize - b->Head)*b->persize,b->Data, b->Head*b->persize);
       
        free(b->Data);
        b->Data=p;
        b->Head=0;
        b->Tail=b->MaxSize;
        b->MaxSize=b->MaxSize*2;
      }

      memcpy(b->Data + b->persize * b->Tail, new_data_addr, new_data_size);
      memcpy(b->Data + b->persize * b->Tail + new_data_size, &b->FanOut, INTSIZE);//改进点：改为赋值

      b->Tail = (b->Tail + 1) % b->MaxSize; // 还需要取模

      b->Data_Count ++;
      b->Sum ++;                  //跟新数据
      DF_FN *temp_f;             //用于取目标fn
      DF_Ready *temp_r, *new_r;
      int done = 0;
      int flag;
      int finish_data_count;
      int finish_flag;
      for (int i=0; i < b->FanOut; i++) {
        done = 0;
        new_r = NULL;

        temp_f = b->DF_Fun_Index[i];
        // printf("寫入數據111\n");
        // printf("%d\n", temp_f);
        pthread_mutex_lock(&temp_f->ready_lock);
        // printf("寫入數據\n");
        finish_data_count = temp_f->DF_Fun_AD_InPut->TargetNum; // 当前方法需要多少个数据
        finish_flag = (1 << finish_data_count) - 1; // 2 => 0000 0011
        flag = b->DF_flag_Index[i]; // 当前方法的第几个数据
        temp_r = temp_f->ready;
        while(temp_r) {
          if ((temp_r->Flags >> flag) & 1) {     //第几个数据组的第flag个数据已就绪，则看下一组
            temp_r = temp_r->next;
            continue;
          }
          else {
            temp_r->Flags = temp_r->Flags | (1 << flag);
            done = 1;
            break;
          }
        }

        temp_r = temp_f->ready;
        if (!done) {           //最后一个ready为NULL
          new_r = (DF_Ready*)malloc(sizeof(DF_Ready));
          new_r->Flags = 1 << flag;
          new_r->next = 0;
          if (temp_r) {
            while(temp_r->next){
              temp_r = temp_r->next;
            }
            temp_r->next = new_r;
          } 
          else {
                   temp_f->ready = new_r;
          }
               // add to tail
        }

        // judge that whether the function can run
        temp_r = temp_f->ready;
        if(temp_r->Flags == finish_flag) {
          // if now flag == finish
          // fixme: not consider that if the queue is full
          assert (threadpool_add(table->pool, (void (*)(void *))(temp_f->Func), NULL, temp_f->item_index)==0);
          #ifdef DEBUG
            printf_thread_info(table);
          #endif
          // consider that it must full in head, so if fail, all fail
          temp_f->ready = temp_r->next;
          free(temp_r);
        }
        pthread_mutex_unlock(&temp_f->ready_lock);
      }
      pthread_rwlock_unlock(&b->lock);

      #ifdef DEBUG
        printf_thread_info(table);
      #endif
    }

  }

  pthread_rwlock_unlock(&F->realfinish_lock);
  va_end(ap);
}

/*
 * Method: DF_AD_GetData
 * Input:
 *   - F: DF_FN, may get function data 
 *   - ...:
 *     - Data: void?, may a address link to ad
 *     - Size: Data size
 *
 * */
int DF_AD_GetData(DF_FN* F, ...) {      //void 修改为返回当前是A函数的第几次运行
  va_list ap;
  va_start(ap, F);

  void *data_addr;
  int data_size;

  int data_count = (F->DF_Fun_AD_InPut->TargetNum);

  DF_AD *temp_ad;
  int rest_count;
  pthread_rwlock_wrlock(&F->finish_lock);
  int finish_num = F->FinishNum;
  F->FinishNum ++;
   

  for (int i=0; i<data_count; i++) {
    data_addr = va_arg(ap, void*);
    data_size = va_arg(ap, int);

    temp_ad = F->DF_Fun_AD_InPut->Target[i];
    pthread_rwlock_wrlock(&temp_ad->lock);//改动：锁的方式

    memcpy(data_addr, temp_ad->Data + ((finish_num-temp_ad->Order+temp_ad->Head) % temp_ad->MaxSize) * 
      temp_ad->persize, data_size); // copy data

    // should make sure that temp_ad's data only operated by one thread

    // update list start
    memcpy(&rest_count, temp_ad->Data + ((finish_num-temp_ad->Order+temp_ad->Head) % temp_ad->MaxSize) * temp_ad->persize + data_size, INTSIZE); //修改点： 改为赋值
    //pthread_rwlock_unlock(&temp_ad->lock);//改动：锁的方式
    //pthread_rwlock_wrlock(&temp_ad->lock);//改动：锁的方式
    if (rest_count > 1) {
    // rest more than 1, multi it
      rest_count = rest_count - 1;
      memcpy(temp_ad->Data + ((finish_num-temp_ad->Order+temp_ad->Head) % temp_ad->MaxSize) * temp_ad->persize + data_size, &rest_count, INTSIZE);
    } else {
    // rest is one, clean
      temp_ad->Head = (temp_ad->Head + 1) % temp_ad->MaxSize;
      temp_ad->Data_Count--;
      temp_ad->Order++;
    }
    pthread_rwlock_unlock(&temp_ad->lock);
    // update list end
  }
  pthread_rwlock_unlock(&F->finish_lock);
  va_end(ap);
  finish_num++;
  return finish_num;
}

void DF_SOURCE_Get_And_Update(DF_FN* F, int* count) {
  pthread_rwlock_wrlock(&F->finish_lock);
  F->FinishNum ++;
  *count = F->FinishNum;
  pthread_rwlock_unlock(&F->finish_lock);
}

void DF_Init(DF_TFL* table, int InputFNNum, ...) {
  va_list ap;
  va_start(ap, InputFNNum);

  table->Num = InputFNNum;
  table->Target = (DF_FN**)malloc(sizeof(DF_FN*) * InputFNNum);
  table->pool = NULL;

  table->Func_Target = (void (**) (void *))malloc(sizeof(void (*) (void *)) * InputFNNum);
  table->should_hash = 1;

  for (int i=0; i<InputFNNum; i++) {
    table->Target[i] = va_arg(ap, DF_FN*);
    table->Target[i]->stop = 0;//改动
    table->Target[i]->item_index = i;
    table->Func_Target[i] = (void (*) (void *))table->Target[i]->Func;
  } 

  va_end(ap);
}

void DF_Thread_Init(DF_TFL* table, int threadnum, int queuesize) {
  if (!table->pool){
    table->thread_num = threadnum;
    /* 断言线程池创建成功 */
    assert((table->pool = threadpool_create(threadnum, queuesize, 0)) != NULL);
  }
}

void DF_SourceInit(DF_TFL *table, int sourcenum, ...) {
  va_list ap;
  va_start(ap, sourcenum);
  DF_FN* source_fn_addr;

  table->source_list_len = sourcenum;
  table->source_list = (DF_SI*)malloc(sizeof(DF_SI) * sourcenum);
  table->item_index_order = (int*)malloc(sizeof(int) * sourcenum);

  for (int i=0; i<sourcenum; i++) {
    source_fn_addr = va_arg(ap, DF_FN*);
    table->source_list[i].F = source_fn_addr;
    table->source_list[i].stop = 1; //修改默认停止运行,调用DF_run时会改为允许运行
    table->source_list[i].count=0; // 临时 ****用于源数据******
    table->item_index_order[i] = i;
  }
}

void DF_OutputInit(DF_TFL *table, int outputnum, ...) {
  va_list ap;
  va_start(ap, outputnum);

  table->output_list_len = outputnum;
  table->output_list = (DF_AD**)malloc(sizeof(DF_AD*) * outputnum);

  for (int i=0; i<outputnum; i++) {
      table->output_list[i] = va_arg(ap, DF_AD*);
  }
}

int DF_Should_Stop(DF_TFL* table) {
  // queue is 0
  // in run count is 0 (mean that not task run)
  // all source is stop
  int all_stop = 1;
  for (int i=0; i < table->source_list_len; i++) {
    if(table->source_list[i].stop == 0) {
      all_stop = 0;
      break;
    }
  }
  return threadpool_is_idle(table->pool)  && all_stop;
}

void printf_thread_info(DF_TFL* table) {
  system("clear");

  for (int i=0; i<table->thread_num; i++) {
    if (table->thread_task[i] != -1)
      c_printf("Thread: %d, [g]%s\n", i, table->Target[table->thread_task[i]]->Name);
    else
      c_printf("Thread: %d, [r]%s\n", i, "NULL");
    }

  for (int i=0; i<table->Num; i++) {
    c_printf("Func: [m]%s, Count: [m]%d\n", table->Target[i]->Name, table->Target[i]->FinishNum);
  }

  int count;
  int** index;
  index = (int**)malloc(sizeof(int*));
  queue_info(table->pool, &count, index);

  for (int i=0; i<count; i++) {
    c_printf("Queue: %d, [c]%s\n", i, table->Target[(*index)[i]]->Name);

  }
}

void DF_Loop(DF_TFL* table) {
  // will replace with get from a list to use many source
  int queue_count;
  table->thread_task = get_thread_info_addr(table->pool);
  int* count_list;
  count_list = (int*)malloc(sizeof(int) * table->Num);
  #ifdef DEBUG
    int begin, end;
    begin = time(NULL);
  #endif
  while(1) {
    if (DF_Should_Stop(table)) {
      break;
    }
    queue_count = threadpool_queue_count(table->pool);

    if (queue_count < table->source_list_len) {           //队列小于源函数数目  
      for (int i=0; i<table->source_list_len; i++) {
        if (!table->source_list[i].stop) {
          assert (threadpool_add(table->pool, (void (*)(void *))(table->source_list[i].F->Func), 
            NULL, table->source_list[i].F->item_index)==0);

          table->source_list[i].count --;            //临时 ****用于源数据******
          if(table->source_list[i].count==0){
            table->source_list[i].stop = 1;
          }
          #ifdef DEBUG
            printf_thread_info(table);
          #endif
        }
      }
    }
       /*
        if (queue_count > 2) {
            for(int i=0; i<table->Num; i++) {
                count_list[i] = 0;
            }
            for(int i=0; i<table->thread_num; i++) {
                if (table->thread_task[i] == -1)
                    continue;
                count_list[table->thread_task[i]] ++;
            }
            for(int i=0; i<table->Num; i++) {
                int min_index = 0;
                for(int j=1; j<table->Num; j++) {
                    if (count_list[j] < count_list[min_index]) {
                        min_index = j;
                    }
                }
                count_list[min_index] = 999999;
                table->item_index_order[i] = min_index;
            }
            
            // 性能点，调度任务应该尽量少执行
            // 尝试每完成一圈任务调度一次？
            // 目前的频度比较高
           // if (table->Num > 3)
              //  order_by_item_and_hash(table->pool, table->Func_Target, table->item_index_order, 
              table->source_list_len, table->should_hash);

        }
           */
  }
  #ifdef DEBUG
    end = time(NULL);
    printf("time=%d\n", end - begin);
  #endif
}

void DF_Update_output(DF_TFL* table) {
  table->result = (void**)malloc(sizeof(void*) * table->output_list_len);
  DF_AD* temp_ad;

  for (int i = 0; i < table->output_list_len; i++) {
    temp_ad = table->output_list[i];
    int data_count = temp_ad->Data_Count;
    int data_size = temp_ad->persize - INTSIZE;
    table->result[i] = (void*)malloc(data_size * data_count);
    for (int j = 0; j < data_count; j++) {
      memcpy(table->result[i] + (j * data_size), 
        temp_ad->Data + ((temp_ad->Head + j) * temp_ad->persize), data_size);
    }
  }
}

void DF_Destory_Ready(DF_FN* fun_node) {
  DF_Ready* temp_ready;
  DF_Ready* save;
  temp_ready = fun_node->ready;

  while(temp_ready) {
    save = temp_ready->next;
    free(temp_ready);
    temp_ready = save;
  }
}

void DF_Destory_AD(DF_TADL* output) {
  int Num = output->TargetNum;
  DF_AD* temp_ad;
  for (int i=0; i<Num; i++) {
    temp_ad = output->Target[i];
    free(temp_ad->Data);
    free(temp_ad->DF_Fun_Index);
    free(temp_ad->DF_flag_Index);
  }
}

void DF_Destory_TADL(DF_FN* fun_node) {
  free(fun_node->DF_Fun_AD_InPut);
  DF_Destory_AD(fun_node->DF_Fun_AD_OutPut);
  free(fun_node->DF_Fun_AD_OutPut);
}

void DF_Destory_FN(DF_TFL* table) {
  DF_FN* temp_fn;
  for (int i=0; i<table->Num; i++) {
    temp_fn = table->Target[i];
    DF_Destory_Ready(temp_fn);
    DF_Destory_TADL(temp_fn);
  }
  free(table->Target);
}

void DF_Destory_source(DF_TFL* table) {
  free(table->source_list);
}

void DF_Destory_TFL(DF_TFL* table) {
  threadpool_destroy(table->pool, 0);
  free(table->output_list);
  free(table->Func_Target);
  free(table->item_index_order);
  free(table->thread_task);
}

void DF_Destory_And_Update_Final_Data(DF_TFL* table) {

  // update to output_list
  DF_Update_output(table);

  // free FN
  DF_Destory_FN(table);

  // free source
  DF_Destory_source(table);

  // free TFL
  DF_Destory_TFL(table);
}

void** DF_Result(DF_TFL *table) {
  return table->result;
}

void DF_Run (DF_TFL *table, int DF_Count) {
  //DF_Thread_Init(table, get_nprocs() * 2, 64);
  #ifdef THREADNUM
    DF_Thread_Init(table, THREADNUM, THREADNUM * 15+2000);
  #else
    DF_Thread_Init(table, 16, 200);
  #endif
  //改动
  if(DF_Count)
    for(int i=0; i<table->source_list_len; i++)
    {
      table->source_list[i].count = DF_Count;
      table->source_list[i].stop = 0;
      table->source_list[i].F->stop = 0;
    }
  else//在运行过程停止
    for(int i=0; i<table->source_list_len; i++)
    {
      table->source_list[i].count = 0;
      table->source_list[i].stop = 0;
      table->source_list[i].F->stop = 0;
    }    
  DF_Loop(table);
  // DF_Source_Init(source_data_addr, datasize, elementcount); // fixed by user
  DF_Destory_And_Update_Final_Data(table);
}

void DF_Source_Stop(DF_TFL *table, int item_index) {
  // clean_index_in_queue(table->pool, item_index);
  table->source_list[item_index].stop = 1;
  table->source_list[item_index].F->stop = 1;
}

//wcy
int use_funcname_to_get_item_index(DF_TFL *table,char* funcname)
{
  for(int i=0;i<table->Num;i++)
  {
    if(strcmp(table->source_list[i].F->Name,funcname)==0)
      return i;
  }
  return -1;
}

