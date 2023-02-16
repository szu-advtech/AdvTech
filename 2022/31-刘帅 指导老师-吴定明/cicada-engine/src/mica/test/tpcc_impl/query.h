#pragma once 

#include "global.h"

class workload;
class tpcc_query;

class base_query {
 public:
  virtual void init(uint64_t thd_id, workload * h_wl) = 0;
};

// All the querise for a particular thread.
class Query_thd {
 public:
  void init(workload * h_wl, int thread_id);
  base_query * get_next_query(); 

  int q_idx;
  tpcc_query * queries;

  char pad[CL_SIZE - sizeof(void *) - sizeof(int)];
  drand48_data buffer;
};

// TODO we assume a separate task queue for each thread in order to avoid 
// contention in a centralized query queue. In reality, more sofisticated 
// queue model might be implemented.
class Query_queue {
 public:
  void init(workload * h_wl);
  void init_per_thread(int thread_id);
  base_query * get_next_query(uint64_t thd_id); 

 private:
  static void * threadInitQuery(void * This);

  Query_thd ** all_queries;
  workload * _wl;
  static int _next_tid;
};

//==========================================================================
//IMPLEMENT

void Query_thd::init(workload * h_wl, int thread_id) {
}
base_query * Query_thd::get_next_query() {
  return nullptr;
}

void Query_queue::init(workload * h_wl) {
}
void Query_queue::init_per_thread(int thread_id) {
}
base_query * Query_queue::get_next_query(uint64_t thd_id) {
  return nullptr;
}
void * Query_queue::threadInitQuery(void * This) {
  return nullptr;
}

