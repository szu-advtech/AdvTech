#include "tpcc_impl/tpcc.h"
#include "tpcc_impl/query.h"
#include "tpcc_impl/thread.h"
#include <thread>
#include <vector>

#if defined (RECOVERY)
template <class StaticConfig>
class ThreadInfoTemplate {
 public:
  std::mutex mtx;
  pthread_t first;
  uint16_t thread_cnt;
  uint16_t active_cnt;
  DB *db;
  pthread_t pids[StaticConfig::kMaxLCoreCount];
 public:
  ThreadInfoTemplate () {
    first = 0;
    thread_cnt = 0;
    active_cnt = 0;
    db = nullptr;
    for (uint16_t i=0; i< StaticConfig::kMaxLCoreCount; i++) {
      pids[i] = 0;
    }
  }
};

typedef ThreadInfoTemplate<DBConfig> ThreadInfo;
static ThreadInfo thread_info;

void process_handle_sigint (int sig) {
  assert (sigint == SIGINT);
  (void) sig;
  printf ("process handle sigint \n");
  thread_info.mtx.lock();
  if (thread_info.first == 0) {
    thread_info.first = pthread_self ();
    thread_info.mtx.unlock ();
    printf ("process pthread_kill thread_info.thread_cnt=%d\n", 
      thread_info.thread_cnt);
    for (auto i=0; i< thread_info.thread_cnt; i++) {
      if (thread_info.pids[i] == thread_info.first) {
        continue;
      }
      printf ("process thread=%lu pthread_kill thread=%lu\n", 
        thread_info.first, thread_info.pids[i]);
      pthread_kill (thread_info.pids[i], SIGINT);
    }
    printf ("process thread=%lu first wait\n", pthread_self());
    while (1) {
      uint16_t l_active_cnt = 0;
      thread_info.mtx.lock ();
      l_active_cnt = thread_info.active_cnt;
      thread_info.mtx.unlock ();
      if (l_active_cnt > 1) {
        std::this_thread::yield(); // wait other thread exit
      }
      else {
        // assert (l_active_cnt == 1);
        break;
      }
    }
    printf ("process thread=%lu first recovery\n", pthread_self());
    thread_info.db->recovery_with_multiple_thread (kRecoveryThread);
    exit (1);
  }
  else {
    thread_info.active_cnt--;
    thread_info.mtx.unlock ();
    printf ("process thread=%lu pthread_exit\n", pthread_self());
    pthread_exit (nullptr);
  }
  thread_info.mtx.unlock();
}

void daemon_handle_sigint (int sig, siginfo_t *siginfo, void *ctx) {
  assert (sig == SIGINT);
  (void) sig;
  (void) ctx;
  printf ("daemon handle sigint \n");
  kill (siginfo->si_int, SIGINT);
}

int daemon () {
  printf ("daemon running pid=%d\n", getpid());
  pid_t pid1 = wait (nullptr);  // wait for sigint
  printf ("wait 1 pass pid=%d\n", pid1);
  int status;
  pid_t pid2 = wait (&status);  // wait for end of process 
  printf ("wait 2 pass daemon end pid=%d status=%d\n", pid2, status);
  if(WIFEXITED(status))//WIFEXITED宏的释义： wait if exit ed
  {
    printf("son process return code %d\n",WEXITSTATUS(status));
  }else if(WIFSIGNALED(status)) {
    printf("son process interupt code: %d\n",WTERMSIG(status));
  }else if(WIFSTOPPED(status)) {
    printf("son process suspend code: %d\n",WSTOPSIG(status));
  }else {
    printf("other\n");
  }
  return 0;
}
#endif

void* f(void*);
thread_t** m_thds;
void* f(void* id) {
  uint64_t tid = (uint64_t)id;
  m_thds[tid]->run();
  return nullptr;
}

#if defined (RECOVERY)
int process(int argc, const char* argv[]) {
  printf ("process running pid=%d\n", getpid());
#else
int main(int argc, const char* argv[]) {
#endif

  if (argc != 4) {
    printf(
        "%s WH_NUM TX-COUNT TD-COUNT\n",
        argv[0]);
    return EXIT_FAILURE;
  }

  uint64_t wh_num   = static_cast<uint64_t>(atol(argv[1]));
  uint64_t tx_count = static_cast<uint64_t>(atol(argv[2]));
  uint64_t td_count = static_cast<uint64_t>(atol(argv[3]));

  g_thread_cnt = td_count;
  g_num_wh     = wh_num;
  g_transaction_cnt = tx_count;

  printf("wh_num   = %" PRIu64 "\n", wh_num);
  printf("tx_count = %" PRIu64 "\n", tx_count);
  printf("td_count = %" PRIu64 "\n", td_count);
  printf("\n");

#if defined (test)
  printf ("Method: test\n");
#elif defined (CWBL)
  printf ("Method: CWBL\n");
#elif defined (CZEN)
  printf ("Method: CZEN\n");
#else
  printf ("Method: ERROR\n");
#endif

#if defined (RECOVERY)
  thread_info.thread_cnt = static_cast<uint16_t>(td_count);
  thread_info.active_cnt = static_cast<uint16_t>(td_count);
  thread_info.db = &db;
#endif

  workload *m_wl = new tpcc_wl;
  // return 0;

  ::mica::util::lcore.pin_thread(0);
  m_wl->init_mica(td_count);
  printf("mica initialized!\n");

  // return 0;

  m_wl->init ();
  printf("workload initialized!\n");

  return 0;

  {
    vector<thread> threads;
    for (uint16_t thread_id = 0; thread_id < td_count; thread_id++) {
      threads.emplace_back([&, thread_id] {
        ::mica::util::lcore.pin_thread(thread_id);

        m_wl->mica_db->activate(thread_id);
        for (auto it : m_wl->tables) {
          auto table = it.second;
          auto mica_tbl = table;
            uint64_t i = 0;
            mica_tbl->mica_tbl->renew_rows(m_wl->mica_db->context(thread_id), 0, i,
                                 static_cast<uint64_t>(-1), false);
            // printf ("Renewing Table:");
            // mica_tbl->mica_tbl->print_table_status();
        }
        for (auto it : m_wl->hash_indexes) {
          auto index = it.second;
          auto idx = index;

            auto mica_tbl = idx->index_table();
            uint64_t i = 0;
            mica_tbl->renew_rows(m_wl->mica_db->context(thread_id), 0, i,
                                 static_cast<uint64_t>(-1), false);
            // printf ("Renewing HashIndex:");
            // mica_tbl->print_table_status();
        }
        for (auto it : m_wl->ordered_indexes) {
          auto index = it.second;
          auto idx = index;
            auto mica_tbl = idx->index_table();
            uint64_t i = 0;
            mica_tbl->renew_rows(m_wl->mica_db->context(thread_id), 0, i,
                                 static_cast<uint64_t>(-1), false);
            // printf ("Renewing BtreeIndex:");
            // mica_tbl->print_table_status();
        }
        m_wl->mica_db->deactivate(thread_id);
      });
    }
    while (threads.size() > 0) {
      threads.back().join();
      threads.pop_back();
    }
    printf("MICA tables renewed\n");
  }

  printf ("\nDRAM Page Pool:\n");
  m_wl->mica_db->page_pool()->print_status();
  printf ("NVRAM Page Pool:\n");
  m_wl->mica_db->aep_page_pool()->print_status();
  printf ("\n");

#if defined (test) && defined (RECOVERY)
  m_wl->mica_db->make_checkpoint ();
#endif

  printf ("\nDRAM Page Pool:\n");
  m_wl->mica_db->page_pool()->print_status();
  printf ("NVRAM Page Pool:\n");
  m_wl->mica_db->aep_page_pool()->print_status();
  printf ("\n");
 
  // use Alloc to alloc memory
  printf("generating workload\n");
  auto mica_dram = m_wl->mica_alloc;
  uint64_t thd_cnt = td_count;
  pthread_t p_thds[thd_cnt - 1];

  m_thds = new thread_t*[thd_cnt];
  for (uint32_t i = 0; i < thd_cnt; i++) {
    m_thds[i] = (thread_t*) mica_dram->malloc_contiguous(sizeof(thread_t));
  }

  // query_queue should be the last one to be initialized!!!
  // because it collects txn latency
  query_queue = (Query_queue*) mica_dram->malloc_contiguous(sizeof(Query_queue));
  query_queue->init(m_wl);

  pthread_barrier_init(&warmup_bar, nullptr, (unsigned int)td_count);
  printf("query_queue initialized!\n");

  pthread_barrier_init(&start_bar, nullptr, (unsigned int)(td_count + 1));
  for (uint32_t i = 0; i < td_count; i++) m_thds[i]->init(i, m_wl);

  m_wl->mica_db->logger()->enable_logging ();
  m_wl->mica_db->reset_stats();
  m_wl->mica_db->reset_backoff();

  if (WARMUP > 0) {
    printf("WARMUP start!\n");
    for (uint32_t i = 0; i < td_count; i++) {
      uint64_t vid = i;
      pthread_create(&p_thds[i], nullptr, f, (void*)vid);
    }
    pthread_barrier_wait(&start_bar);
    for (uint32_t i = 0; i < td_count; i++) pthread_join(p_thds[i], nullptr);

    printf("WARMUP finished!\n");
  }
  warmup_finish = true;

  m_wl->mica_db->reset_stats();

  for (uint32_t i = 0; i < td_count; i++) {
    uint64_t vid = i;
    pthread_create(&p_thds[i], nullptr, f, (void*)vid);
  }
  pthread_barrier_wait(&start_bar);

  struct timeval starttime, endtime;
  gettimeofday(&starttime, nullptr);
  for (uint32_t i = 0; i < thd_cnt; i++) pthread_join(p_thds[i], nullptr);
  gettimeofday(&endtime, nullptr);

  double t = (double)(endtime.tv_sec - starttime.tv_sec)
           + (double)(endtime.tv_usec - starttime.tv_usec) / 1000000.;

  printf("page_used:       %lu pages\n", 
         m_wl->mica_db->page_pool()->both_page_num());
  printf("memory_size:     %7.3lf GB\n", 
         m_wl->mica_db->page_pool()->both_memory_use());

  printf("aep_page_used:   %lu pages\n", 
         m_wl->mica_db->aep_page_pool()->both_page_num());
  printf("aep_memory_size: %7.3lf GB\n", 
         m_wl->mica_db->aep_page_pool()->both_memory_use());

  printf("aep_clwb_cnt:    %7.3lf M\n",  
         m_wl->mica_db->get_clwb_cnt ());
  printf("aep_sfence_cnt:  %7.3lf M\n",  
         m_wl->mica_db->get_sfence_cnt ());
  printf("aep_persist_size:%7.3lf GB\n", 
         m_wl->mica_db->get_persist_size());

  printf("PASS! TimeUse = %7.3lf seconds\n", t);
  m_wl->mica_db->print_stats(t, t * (double)thd_cnt);
  for (auto it : m_wl->tables) {
    auto table = it.second;
    auto mica_tbl = table;
      printf("table %s :\n", it.first.c_str());
      mica_tbl->mica_tbl->print_table_status();
      // printf("\n");
  }
  m_wl->mica_db->print_pool_status();
  m_wl->mica_page_pools[0]->print_status();
  m_wl->mica_page_pools[1]->print_status();
  ::mica::util::Latency inter_commit_latency;
  for (uint32_t i = 0; i < thd_cnt; i++)
    inter_commit_latency += m_wl->mica_db->context(static_cast<uint16_t>(i))
                                ->inter_commit_latency();

  return 0;
}

#if defined (RECOVERY)
int main(int argc, const char* argv[]) {
  // for recovery, require data initilized as 0 before test
  int ret = system ("rm /mnt/mypmem0/mypmem*");
  (void) ret;

  pid_t fpid = fork ();
  if (fpid < 0) {
    printf ("Error: fork wrong!\n");
    return 0;
  }
  else if (fpid == 0) {
    // son process
    signal (SIGINT, process_handle_sigint);
    process (argc, argv);
  }
  else {
    // parent daemon
    struct sigaction act;
    act.sa_sigaction = daemon_handle_sigint; // sigaction
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_SIGINFO; // declear handle with paramter
    sigaction (SIGINT, &act, nullptr);
    daemon ();
  }
}
#endif

