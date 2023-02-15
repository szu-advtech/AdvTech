#include <cstdio>
#include <thread>
#include <random>

#include <mutex>
#include <signal.h>
#include <sys/wait.h>

#include "mica/util/lcore.h"
#include "mica/util/zipf.h"
#include "mica/util/rand.h"

#include "mica/transaction/db.h"
#include "mica/test/test_tx_conf.h"

typedef DBConfig::Alloc Alloc;
typedef DBConfig::PmemAlloc PmemAlloc;
typedef DBConfig::LoggerInterface LoggerInterface;
typedef DBConfig::Logger Logger;
typedef DBConfig::PmemLogger PmemLogger;
typedef DBConfig::Timestamp Timestamp;
typedef DBConfig::ConcurrentTimestamp ConcurrentTimestamp;
typedef DBConfig::Timing Timing;
typedef ::mica::transaction::PagePool<DBConfig> PagePool;
typedef ::mica::transaction::DB<DBConfig> DB;
typedef ::mica::transaction::Table<DBConfig> Table;
typedef DB::HashIndexUniqueU64 HashIndex;
typedef DB::BTreeIndexUniqueU64 BTreeIndex;
typedef ::mica::transaction::RowVersion<DBConfig> RowVersion;
typedef ::mica::transaction::RowAccessHandle<DBConfig> RowAccessHandle;
typedef ::mica::transaction::RowAccessHandlePeekOnly<DBConfig>
        RowAccessHandlePeekOnly;
typedef ::mica::transaction::PmemRowAccessHandle<DBConfig> PmemRowAccessHandle;
typedef ::mica::transaction::PmemRowAccessHandlePeekOnly<DBConfig>
        PmemRowAccessHandlePeekOnly;
typedef ::mica::transaction::Transaction<DBConfig> Transaction;
typedef ::mica::transaction::PmemTransaction<DBConfig> PmemTransaction;
typedef ::mica::transaction::Result Result;
static ::mica::util::Stopwatch sw;

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
 
/* 
  pid_t pid3 = wait (nullptr);  // wait for sigint
  printf ("wait 3 pass pid=%d\n", pid3);
  pid_t pid4 = wait (nullptr);  // wait for end of process 
  printf ("wait 4 pass daemon end pid=%d\n", pid4);
*/
 
  return 0;
}
#endif

// Worker task.

struct Task {

#if defined (RECOVERY)
  ThreadInfo *thread;
  uint64_t txn_count;
#endif

  DB* db;
  Table* tbl;
  HashIndex* hash_idx;
  BTreeIndex* btree_idx;

  uint64_t thread_id;
  uint64_t num_threads;

  // Workload.
  uint64_t num_rows;
  uint64_t tx_count;
  uint16_t* req_counts;
  uint8_t* read_only_tx;
  uint32_t* scan_lens;
  uint64_t* row_ids;
  uint8_t* column_ids;
  uint8_t* op_types;

  // State (for VerificationLogger).
  uint64_t tx_i;
  uint64_t req_i;
  uint64_t commit_i;

  // Results.
  struct timeval tv_start;
  struct timeval tv_end;

  uint64_t committed;
  uint64_t scanned;

  // Transaction log for verification.
  uint64_t* commit_tx_i;
  Timestamp* commit_ts;
  Timestamp* read_ts;
  Timestamp* write_ts;
} __attribute__((aligned(64)));

template <class StaticConfig>
class VerificationLogger
    : public ::mica::transaction::LoggerInterface<StaticConfig> {
 public:
  VerificationLogger() : tasks(nullptr) {}

  bool log(const ::mica::transaction::Transaction<StaticConfig>* tx) {
    if (tasks == nullptr) return true;

    auto task = &((*tasks)[tx->context()->thread_id()]);
    auto tx_i = task->tx_i;
    auto req_i = task->req_i;
    auto commit_i = task->commit_i;

    task->commit_tx_i[commit_i] = tx_i;
    task->commit_ts[tx_i] = tx->ts();

    // XXX: Assume we have the same row ordering in the read/write set.
    uint64_t req_j = 0;
    for (auto j = 0; j < tx->access_size(); j++) {
      if (tx->accesses()[j].state == ::mica::transaction::RowAccessState::kPeek)
        continue;

      assert(req_j < task->req_counts[tx_i]);

      task->read_ts[req_i + req_j] = tx->accesses()[j].read_rv->wts;
      if (tx->accesses()[j].write_rv != nullptr)
        task->write_ts[req_i + req_j] = tx->accesses()[j].write_rv->wts;
      else
        task->write_ts[req_i + req_j] = task->read_ts[req_i + req_j];
      req_j++;
    }
    assert(req_j == task->req_counts[tx_i]);
    return true;
  }

  std::vector<Task>* tasks;
};

template <class Logger>
void setup_logger(Logger* logger, std::vector<Task>* tasks) {
  (void)logger;
  (void)tasks;
}

template <>
void setup_logger(VerificationLogger<DBConfig>* logger,
                  std::vector<Task>* tasks) {
  logger->tasks = tasks;
}

static volatile uint16_t running_threads;
static volatile uint8_t stopping;

void worker_proc(Task* task) {
  ::mica::util::lcore.pin_thread(static_cast<uint16_t>(task->thread_id));

#if defined (RECOVERY)
  task->thread->pids[task->thread_id] = pthread_self ();
  uint64_t recovery_threshold= task->txn_count - static_cast<uint64_t>(
    1.0/static_cast<double>(kRecoverySignal)*static_cast<double>(task->txn_count));
  printf ("worker thread_id=%lu pthread_self=%lu\n", task->thread_id, pthread_self());
#endif

  auto ctx = task->db->context();
  auto tbl = task->tbl;
  auto hash_idx = task->hash_idx;
  auto btree_idx = task->btree_idx;

  __sync_add_and_fetch(&running_threads, 1);
  while (running_threads < task->num_threads) ::mica::util::pause();

  Timing t(ctx->timing_stack(), &::mica::transaction::Stats::worker);

  gettimeofday(&task->tv_start, nullptr);

  uint64_t next_tx_i = 0;
  uint64_t next_req_i = 0;
  uint64_t commit_i = 0;
  uint64_t scanned = 0;

  task->db->activate(static_cast<uint16_t>(task->thread_id));
  while (task->db->active_thread_count() < task->num_threads) {
    ::mica::util::pause();
    task->db->idle(static_cast<uint16_t>(task->thread_id));
  }

  if (kVerbose) printf("lcore %" PRIu64 "\n", task->thread_id);

#if defined (test) || defined (CWBL)
  Transaction tx(ctx);
#elif defined (CZEN)
  PmemTransaction tx(ctx);
#endif

  while (next_tx_i < task->tx_count && !stopping) {
    uint64_t tx_i;
    uint64_t req_i;

    tx_i = next_tx_i++;
    req_i = next_req_i;
    next_req_i += task->req_counts[tx_i];

    task->tx_i = tx_i;
    task->req_i = req_i;
    task->commit_i = commit_i;

#if defined (RECOVERY)
    if (task->thread_id ==0 && next_tx_i > recovery_threshold) {
      // printf ("worker_proc self=%ld send sigint pid=%d ppid=%d\n", pthread_self(), getpid(), getppid());
      union sigval val;
      val.sival_int = getpid();
      sigqueue(getppid(), SIGINT, val);
    }
#endif

    while (true) {
      bool aborted = false;

      uint64_t v = 0;

      bool use_peek_only = kUseSnapshot && task->read_only_tx[tx_i];

#if defined (test) || defined (CWBL)
      bool ret = tx.begin(use_peek_only);
#elif defined (CZEN)
      bool ret = tx.aep_begin(use_peek_only);
#endif

      assert(ret);
      (void)ret;

      for (uint64_t req_j = 0; req_j < task->req_counts[tx_i]; req_j++) {
        uint64_t row_id = task->row_ids[req_i + req_j];
        uint8_t column_id = task->column_ids[req_i + req_j];
        bool is_read = task->op_types[req_i + req_j] == 0;
        bool is_rmw = task->op_types[req_i + req_j] == 1;

        if (hash_idx != nullptr) {
          auto lookup_result =
              hash_idx->lookup(&tx, row_id, kSkipValidationForIndexAccess,
                               [&row_id](auto& k, auto& v) {
                                 (void)k;
                                 row_id = v;
                                 return false;
                               });
          if (lookup_result != 1 || lookup_result == HashIndex::kHaveToAbort) {
            assert(false);

#if defined (test) || defined (CWBL)
            tx.abort();
#elif defined (CZEN)
            printf ("abort because of hash index lookup!\n");
            tx.aep_abort();
#endif

            aborted = true;
            break;
          }
        } else if (btree_idx != nullptr) {
          auto lookup_result =
              btree_idx->lookup(&tx, row_id, kSkipValidationForIndexAccess,
                                [&row_id](auto& k, auto& v) {
                                  (void)k;
                                  row_id = v;
                                  return false;
                                });
          if (lookup_result != 1 || lookup_result == BTreeIndex::kHaveToAbort) {
            assert(false);

#if defined (test) || defined (CWBL)
            tx.abort();
#elif defined (CZEN)
            printf ("abort because of btree index lookup!\n");
            tx.aep_abort();
#endif

            aborted = true;
            break;
          }
        }

        if (!use_peek_only) {

#if defined (test) || defined (CWBL)
          RowAccessHandle rah(&tx);
          if (is_read) {
            if (!rah.peek_row(tbl, 0, row_id, false, true, false) ||
                !rah.read_row()) {
              tx.abort();
              aborted = true;
              break;
            }

            const char* data =
                rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64)
              v += static_cast<uint64_t>(data[j]);
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
          } else {
            if (is_rmw) {
              if (!rah.peek_row(tbl, 0, row_id, false, true, true) ||
                  !rah.read_row() || !rah.write_row(kDataSize)) {
                tx.abort();
                aborted = true;
                break;
              }
            } else {
              if (!rah.peek_row(tbl, 0, row_id, false, false, true) ||
                  !rah.write_row(kDataSize)) {
                tx.abort();
                aborted = true;
                break;
              }
            }

            char* data =
                rah.data() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64) {
              v += static_cast<uint64_t>(data[j]);
              data[j] = static_cast<char>(v);
            }
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
            data[kColumnSize - 1] = static_cast<char>(v);
          }
#elif defined (CZEN)
          PmemRowAccessHandle rah(&tx);
          if (is_read) {
            if (!rah.aep_peek_row(tbl, 0, row_id, false, true, false) ||
                !rah.aep_read_row()) {

              // printf ("abort because of peek read lookup! 290\n");
 
              tx.aep_abort();
              aborted = true;
              break;
            }

            const char* data =
                rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64)
              v += static_cast<uint64_t>(data[j]);
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
          } else {
            if (is_rmw) {
              if (!rah.aep_peek_row(tbl, 0, row_id, false, true, true) ||
                  !rah.aep_read_row() || !rah.aep_write_row(kDataSize)) {

                // printf ("abort because of peek read write lookup! 307\n");

                tx.aep_abort();
                aborted = true;
                break;
              }
            } else {
              if (!rah.aep_peek_row(tbl, 0, row_id, false, false, true) ||
                  !rah.aep_write_row(kDataSize)) {

                // printf ("abort because of peek write lookup! 317\n");

                tx.aep_abort();
                aborted = true;
                break;
              }
            }

            char* data =
                rah.data() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64) {
              v += static_cast<uint64_t>(data[j]);
              data[j] = static_cast<char>(v);
            }
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
            data[kColumnSize - 1] = static_cast<char>(v);
          }
#endif

        } else {
          if (!kUseScan) {

#if defined (test) || defined (CWBL)
            RowAccessHandlePeekOnly rah(&tx);
            if (!rah.peek_row(tbl, 0, row_id, false, false, false)) {
              tx.abort();
              aborted = true;
              break;
            }
            const char* data =
                rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64)
              v += static_cast<uint64_t>(data[j]);
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
#elif defined (CZEN)
            PmemRowAccessHandlePeekOnly rah(&tx);
            if (!rah.aep_peek_row(tbl, 0, row_id, false, false, false)) {
              tx.abort();
              aborted = true;
              break;
            }
            const char* data =
                rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
            for (uint64_t j = 0; j < kColumnSize; j += 64)
              v += static_cast<uint64_t>(data[j]);
            v += static_cast<uint64_t>(data[kColumnSize - 1]);
#endif

          } else if (!kUseFullTableScan) {

#if defined (test) || defined (CWBL)
            RowAccessHandlePeekOnly rah(&tx);

            uint64_t next_row_id = row_id;
            uint64_t next_next_raw_row_id = task->row_ids[req_i + req_j] + 1;
            if (next_next_raw_row_id == task->num_rows)
              next_next_raw_row_id = 0;

            uint32_t scan_len = task->scan_lens[tx_i];
            for (uint32_t scan_i = 0; scan_i < scan_len; scan_i++) {
              uint64_t this_row_id = next_row_id;

              // TODO: Support btree_idx.
              assert(hash_idx != nullptr);

              // Lookup index for next row.
              auto lookup_result =
                  hash_idx->lookup(&tx, next_next_raw_row_id, true,
                                   [&next_row_id](auto& k, auto& v) {
                                     (void)k;
                                     next_row_id = v;
                                     return false;
                                   });
              if (lookup_result != 1 ||
                  lookup_result == HashIndex::kHaveToAbort) {
                tx.abort();
                aborted = true;
                break;
              }

              // Prefetch index for next next row.
              next_next_raw_row_id++;
              if (next_next_raw_row_id == task->num_rows)
                next_next_raw_row_id = 0;
              hash_idx->prefetch(&tx, next_next_raw_row_id);

              // Prefetch next row.
              rah.prefetch_row(tbl, 0, next_row_id,
                               static_cast<uint64_t>(column_id) * kColumnSize,
                               kColumnSize);

              // Access current row.
              if (!rah.peek_row(tbl, 0, this_row_id, false, false, false)) {
                tx.abort();
                aborted = true;
                break;
              }

              const char* data =
                  rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
              for (uint64_t j = 0; j < kColumnSize; j += 64)
                v += static_cast<uint64_t>(data[j]);
              v += static_cast<uint64_t>(data[kColumnSize - 1]);

              rah.reset();
            }
#elif defined (CZEN)
            PmemRowAccessHandlePeekOnly rah(&tx);

            uint64_t next_row_id = row_id;
            uint64_t next_next_raw_row_id = task->row_ids[req_i + req_j] + 1;
            if (next_next_raw_row_id == task->num_rows)
              next_next_raw_row_id = 0;

            uint32_t scan_len = task->scan_lens[tx_i];
            for (uint32_t scan_i = 0; scan_i < scan_len; scan_i++) {
              uint64_t this_row_id = next_row_id;

              // TODO: Support btree_idx.
              assert(hash_idx != nullptr);

              // Lookup index for next row.
              auto lookup_result =
                  hash_idx->lookup(&tx, next_next_raw_row_id, true,
                                   [&next_row_id](auto& k, auto& v) {
                                     (void)k;
                                     next_row_id = v;
                                     return false;
                                   });
              if (lookup_result != 1 ||
                  lookup_result == HashIndex::kHaveToAbort) {
                tx.aep_abort();
                aborted = true;
                break;
              }

              // Prefetch index for next next row.
              next_next_raw_row_id++;
              if (next_next_raw_row_id == task->num_rows)
                next_next_raw_row_id = 0;
              hash_idx->prefetch(&tx, next_next_raw_row_id);

              // Prefetch next row.
              // rah.prefetch_row(tbl, 0, next_row_id,
              //                  static_cast<uint64_t>(column_id) * kColumnSize,
              //                  kColumnSize);

              // Access current row.

              if (!rah.aep_peek_row(tbl, 0, this_row_id, false, false, false)) {
                tx.aep_abort();
                aborted = true;
                break;
              }

              const char* data =
                  rah.cdata() + static_cast<uint64_t>(column_id) * kColumnSize;
              for (uint64_t j = 0; j < kColumnSize; j += 64)
                v += static_cast<uint64_t>(data[j]);
              v += static_cast<uint64_t>(data[kColumnSize - 1]);

              rah.reset();
            }

#endif
            if (aborted) break;
          } else /*if (kUseFullTableScan)*/ {

#if defined (test) || defined (CWBL)
            if (!tbl->scan(&tx, 0,
                           static_cast<uint64_t>(column_id) * kColumnSize,
                           kColumnSize, [&v, column_id](auto& rah) {
                             const char* data =
                                 rah.cdata() +
                                 static_cast<uint64_t>(column_id) * kColumnSize;
                             for (uint64_t j = 0; j < kColumnSize; j += 64)
                               v += static_cast<uint64_t>(data[j]);
                             v += static_cast<uint64_t>(data[kColumnSize - 1]);
                           })) {
              tx.abort();
              aborted = true;
              break;
            }
#elif defined (CZEN)
            // TODO PmemTable Full Scan
#endif

          }
        }
      }

      if (aborted) continue;

      Result result;

#if defined (test) || defined (CWBL)
      if (!tx.commit(&result)) continue;
#elif defined (CZEN)
      if (!tx.aep_commit(&result)) continue;
#endif

      assert(result == Result::kCommitted);

      commit_i++;
      if (kUseScan && use_peek_only) {
        if (!kUseFullTableScan)
          scanned += task->scan_lens[tx_i];
        else
          scanned += task->num_rows;
      }

      break;
    }
  }

  task->db->deactivate(static_cast<uint16_t>(task->thread_id));

  if (!stopping) stopping = 1;

  task->committed = commit_i;
  task->scanned = scanned;

  gettimeofday(&task->tv_end, nullptr);
}

#if defined (RECOVERY)
int process(int argc, const char* argv[]) {
  printf ("process running pid=%d\n", getpid());
#else
int main(int argc, const char* argv[]) {
#endif

  // printf ("RowHead=%lu\n", sizeof(::mica::transaction::RowHead<DBConfig>));
  // return 0;

#if defined (test) || defined (CWBL)
  if (argc != 7 && argc != 10) {
    printf(
        "%s NUM-ROWS REQS-PER-TX READ-RATIO ZIPF-THETA TX-COUNT "
        "THREAD-COUNT [optional] kDataSize kPreMallocSize kPmemPreMallocSize\n",
        argv[0]);
    return EXIT_FAILURE;
  }
#elif defined (CZEN)
  if (argc != 7 && argc != 11) {
    printf(
        "%s NUM-ROWS REQS-PER-TX READ-RATIO ZIPF-THETA TX-COUNT "
        "THREAD-COUNT [optional] kDataSize kPreMallocSize kPmemPreMallocSize Cachenum\n",
        argv[0]);
    return EXIT_FAILURE;
  }
#endif 

  uint64_t num_rows = static_cast<uint64_t>(atol(argv[1]));
  uint64_t reqs_per_tx = static_cast<uint64_t>(atol(argv[2]));
  double read_ratio = atof(argv[3]);
  double zipf_theta = atof(argv[4]);
  uint64_t tx_count = static_cast<uint64_t>(atol(argv[5]));
  uint64_t num_threads = static_cast<uint64_t>(atol(argv[6]));

#if defined (test) || defined (CWBL)
  if (argc == 10) {
    kDataSize = static_cast<uint64_t>(atol(argv[7]));
    kPreMallocSize = static_cast<uint64_t>(atol(argv[8]));
    kPmemPreMallocSize = static_cast<uint64_t>(atol(argv[9]));
  }
#elif defined (CZEN)
  if (argc == 11) {
    kDataSize = static_cast<uint64_t>(atol(argv[7]));
    kPreMallocSize = static_cast<uint64_t>(atol(argv[8]));
    kPmemPreMallocSize = static_cast<uint64_t>(atol(argv[9]));
    kZenCacheSize = static_cast<uint64_t>(atol(argv[10]));
  }
#endif

  Alloc alloc(kNumaInUse, kPreMallocSize);
  PmemAlloc aep_alloc(kNumaInUse, kPmemPreMallocSize);

  // do this to malloc huge page memory or aep-memory
  alloc.pre_malloc_2mb ();
  aep_alloc.pre_malloc_2mb ();

  alloc.print ();
  aep_alloc.print ();

  // alloc.~Alloc();
  // aep_alloc.~PmemAlloc();
  // return 0;

  PagePool* page_pools[2];
  // if (num_threads == 1) {
  //   page_pools[0] = new PagePool(&alloc, page_pool_size, 0);
  //   page_pools[1] = nullptr;
  // } else {
  page_pools[0] = new PagePool(&alloc, &aep_alloc, 0);
  page_pools[1] = new PagePool(&alloc, &aep_alloc, 1);

  printf ("pagepool    = %p aep_only=%u\n",  page_pools[0], page_pools[0]->aep_only());
  printf ("aeppagepool = %p aep_only=%u\n",page_pools[1], page_pools[1]->aep_only());

  printf ("RowHeadSize    =%lu\n", sizeof(::mica::transaction::RowHead<DBConfig>));
  printf ("RowVersionSize =%lu\n", sizeof(::mica::transaction::RowVersion<DBConfig>));
  printf ("RowCommonSize  =%lu\n", sizeof(::mica::transaction::RowCommon<DBConfig>));

  // return 0;

  ::mica::util::lcore.pin_thread(0);

  sw.init_start();
  sw.init_end();

  if (num_rows == 0) {
    num_rows = SYNTH_TABLE_SIZE;
    reqs_per_tx = REQ_PER_QUERY;
    read_ratio = READ_PERC;
    zipf_theta = ZIPF_THETA;
    tx_count = MAX_TXN_PER_PART;
    num_threads = THREAD_CNT;
#ifndef NDEBUG
    printf("!NDEBUG\n");
    return EXIT_FAILURE;
#endif
  }

  printf("num_rows = %" PRIu64 "\n", num_rows);
  printf("reqs_per_tx = %" PRIu64 "\n", reqs_per_tx);
  printf("read_ratio = %lf\n", read_ratio);
  printf("zipf_theta = %lf\n", zipf_theta);
  printf("tx_count = %" PRIu64 "\n", tx_count);
  printf("num_threads = %" PRIu64 "\n", num_threads);
#ifndef NDEBUG
  printf("!NDEBUG\n");
#endif
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

#if defined (test) || defined (CWBL)
  PmemLogger logger(page_pools[1]);
#elif defined (CZEN)
  Logger logger;
#endif

  logger.set_logging_thread_count ((uint16_t)num_threads);
  logger.disable_logging ();
  DB db(page_pools, &logger, &sw, static_cast<uint16_t>(num_threads), kRecoveryThread);

#if defined (RECOVERY)
  thread_info.thread_cnt = static_cast<uint16_t>(num_threads);
  thread_info.active_cnt = static_cast<uint16_t>(num_threads);
  thread_info.db = &db;
#endif

  const bool kVerify =
      typeid(typename DBConfig::Logger) == typeid(VerificationLogger<DBConfig>);

  const uint64_t data_sizes[] = {kDataSize};
  kColumnSize = kDataSize/MICA_COLUMN_COUNT;

#if defined (test) || defined (CWBL)
  bool ret = db.create_table("main", 1, data_sizes, num_rows);
#elif defined (CZEN)
  bool ret = db.create_table("main", 1, data_sizes, num_rows, kZenCacheSize);
#endif

  assert(ret);
  (void)ret;

  printf ("\nDRAM Page Pool:\n");
  db.page_pool()->print_status();
  printf ("NVRAM Page Pool:\n");
  db.aep_page_pool()->print_status();
  printf ("\n");

  auto tbl = db.get_table("main");
  db.activate(0);

  HashIndex* hash_idx = nullptr;
  if (kUseHashIndex) {
    bool ret = db.create_hash_index_unique_u64("main_idx", tbl, num_rows);
    assert(ret);
    (void)ret;

    hash_idx = db.get_hash_index_unique_u64("main_idx");
    Transaction tx(db.context(0));
    hash_idx->init(&tx);
  }

  BTreeIndex* btree_idx = nullptr;
  if (kUseBTreeIndex) {
    bool ret = db.create_btree_index_unique_u64("main_idx", tbl);
    assert(ret);
    (void)ret;

    btree_idx = db.get_btree_index_unique_u64("main_idx");
    Transaction tx(db.context(0));
    btree_idx->init(&tx);
  }

  printf ("\nDRAM Page Pool:\n");
  db.page_pool()->print_status();
  printf ("NVRAM Page Pool:\n");
  db.aep_page_pool()->print_status();
  printf ("\n");
  printf ("Table:\n");
  tbl->print_table_status ();
  printf ("HashIndex:\n");
  hash_idx->print_hash_status ();
  printf ("RowVersionPool(Pmem):\n");
  db.print_pool_status ();
  // return 0;

  {
    printf("initializing table\n");

    std::vector<std::thread> threads;
    uint64_t init_num_threads = std::min(uint64_t(16), num_threads);

    for (uint64_t thread_id = 0; thread_id < init_num_threads; thread_id++) {
      threads.emplace_back([&, thread_id] {
        ::mica::util::lcore.pin_thread(thread_id);

        db.activate(static_cast<uint16_t>(thread_id));
        while (db.active_thread_count() < init_num_threads) {
          ::mica::util::pause();
          db.idle(static_cast<uint16_t>(thread_id));
        }

        // Randomize the data layout by shuffling row insert order.
        std::mt19937 g(thread_id);
        std::vector<uint64_t> row_ids;
        row_ids.reserve((num_rows + init_num_threads - 1) / init_num_threads);
        for (uint64_t i = thread_id; i < num_rows; i += init_num_threads)
          row_ids.push_back(i);
        std::shuffle(row_ids.begin(), row_ids.end(), g);

#if defined (test) || defined (CWBL)
        Transaction tx(db.context(static_cast<uint16_t>(thread_id)));
#elif defined (CZEN)
        PmemTransaction tx(db.context(static_cast<uint16_t>(thread_id)));
#endif

        const uint64_t kBatchSize = 16;
        for (uint64_t i = 0; i < row_ids.size(); i += kBatchSize) {
          while (true) {
#if defined (test) || defined (CWBL)
            bool ret = tx.begin();
#elif defined (CZEN)
            bool ret = tx.aep_begin();
#endif
            if (!ret) {
              printf("failed to start a transaction\n");
              continue;
            }

            bool aborted = false;
            auto i_end = std::min(i + kBatchSize, row_ids.size());
            for (uint64_t j = i; j < i_end; j++) {

#if defined (test) || defined (CWBL)
              RowAccessHandle rah(&tx);
              if (!rah.new_row(tbl, 0, Transaction::kNewRowID, true,
                               kDataSize)) {
                // printf("failed to insert rows at new_row(), row = %" PRIu64
                //        "\n",
                //        j);
                aborted = true;
                tx.abort();
                break;
              }

              //debugforindexalign
              rah.row_access_item()->head->pkey= row_ids[j];

              if (kUseHashIndex) {
                auto ret = hash_idx->insert(&tx, row_ids[j], rah.row_id());
                if (ret != 1 || ret == HashIndex::kHaveToAbort) {
                  // printf("failed to update index row = %" PRIu64 "\n", j);
                  aborted = true;
                  tx.abort();
                  break;
                }
              }
              if (kUseBTreeIndex) {
                auto ret = btree_idx->insert(&tx, row_ids[j], rah.row_id());
                if (ret != 1 || ret == BTreeIndex::kHaveToAbort) {
                  // printf("failed to update index row = %" PRIu64 "\n", j);
                  aborted = true;
                  tx.abort();
                  break;
                }
              }
#elif defined (CZEN)
              PmemRowAccessHandle rah(&tx);
              if (!rah.aep_new_row(tbl, 0, Transaction::kNewRowID, row_ids[j], true,
                                   kDataSize)) {
                // printf("failed to insert rows at new_row(), row = %" PRIu64
                //        "\n",
                //        j);
                aborted = true;
                tx.aep_abort();
                break;
              }

              if (kUseHashIndex) {
                auto ret = hash_idx->insert(&tx, row_ids[j], rah.row_id());
                if (ret != 1 || ret == HashIndex::kHaveToAbort) {
                  // printf("failed to update index row = %" PRIu64 "\n", j);
                  aborted = true;
                  tx.aep_abort();
                  break;
                }
              }
              if (kUseBTreeIndex) {
                auto ret = btree_idx->insert(&tx, row_ids[j], rah.row_id());
                if (ret != 1 || ret == BTreeIndex::kHaveToAbort) {
                  // printf("failed to update index row = %" PRIu64 "\n", j);
                  aborted = true;
                  tx.aep_abort();
                  break;
                }
              }
              // printf ("init insert: key=%lu val=%lu\n", row_ids[j],rah.row_id());
#endif
            }

            if (aborted) continue;
            Result result;

#if defined (test) || defined (CWBL)
            if (!tx.commit(&result)) {
#elif defined (CZEN)
            if (!tx.aep_commit(&result)) {
#endif
              // printf("failed to insert rows at commit(), row = %" PRIu64
              //        "; result=%d\n",
              //        i_end - 1, static_cast<int>(result));
              continue;
            }
            break;
          }
        }


/*       
        // check the table is initialized as think
        {
          printf ("check initilize process \n");
          for (uint64_t i=0; i< num_rows; i++) {
            PmemTransaction tx(db.context(static_cast<uint16_t>(0)));
            tx.aep_begin();
            uint64_t vrow_id;
            auto lookup_result =
              hash_idx->lookup(&tx, i, kSkipValidationForIndexAccess,
                               [&vrow_id](auto& k, auto& v) {
                                 (void)k;
                                 vrow_id = v;
                                 return false;
                               });
            if (lookup_result != 1 || lookup_result == HashIndex::kHaveToAbort) {
              assert(false);
            }

            // printf ("key=%lu\tval=%lu\n", i, vrow_id); 
 
            PmemRowAccessHandle rah(&tx);

            if (!rah.aep_peek_row(tbl, 0, vrow_id, false, true, false) ||
                !rah.aep_read_row()) {

              // printf ("abort because of peek read lookup! key=%lu\tval=%lu\n", i, vrow_id);
              tx.aep_abort();
              break;
            }

            printf ("key=%lu\tval=%lu\tval=%p\trah.row_id()=%lu\thead->primkey=%lu\tread_rv=%p\twrite_rv=%p\taep_rv=%p\taep_rv->pkey=%lu\n",
                     i, 
                     vrow_id,
                     reinterpret_cast<void*>(vrow_id),
                     rah.row_id(),
                     tbl->head(0, rah.row_id())->pkey,
                     rah.row_access_item()->read_rv,
                     rah.row_access_item()->write_rv,
                     rah.row_access_item()->write_rv?
                       rah.row_access_item()->write_rv->aep_rv:
                       rah.row_access_item()->read_rv->aep_rv,
                     rah.row_access_item()->write_rv?
                       rah.row_access_item()->write_rv->aep_rv->pkey:
                       rah.row_access_item()->read_rv->aep_rv->pkey);

            tx.aep_commit();
          }
        }
*/

        db.deactivate(static_cast<uint16_t>(thread_id));
        return 0;
      });
    }

    while (threads.size() > 0) {
      threads.back().join();
      threads.pop_back();
    }

    // TODO: Use multiple threads to renew rows for more balanced memory access.

    printf ("start renewing tables!\n");
    fflush (stdout);

    db.activate(0);
    {
      uint64_t i = 0;
      tbl->renew_rows(db.context(0), 0, i, static_cast<uint64_t>(-1), false);
    }
    if (hash_idx != nullptr) {
      uint64_t i = 0;
      hash_idx->index_table()->renew_rows(db.context(0), 0, i,
                                          static_cast<uint64_t>(-1), false);
    }
    if (btree_idx != nullptr) {
      uint64_t i = 0;
      btree_idx->index_table()->renew_rows(db.context(0), 0, i,
                                           static_cast<uint64_t>(-1), false);
    }
    db.deactivate(0);

    printf ("finish initialized!\n");
    fflush (stdout);

    db.reset_stats();
    db.reset_backoff();

#if defined (test) // generate checkpoint for test
    #if defined (RECOVERY)
    db.make_checkpoint ();
    #endif
#endif
  }

  // tbl->print_table_status ();
  // return 0;

  printf ("\nDRAM Page Pool:\n");
  db.page_pool()->print_status();
  printf ("NVRAM Page Pool:\n");
  db.aep_page_pool()->print_status();
  printf ("\n");
 
  std::vector<Task> tasks(num_threads);
  setup_logger(&logger, &tasks);
  {
    printf("generating workload\n");

    for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
      tasks[thread_id].thread_id = static_cast<uint16_t>(thread_id);
      tasks[thread_id].num_threads = num_threads;
      tasks[thread_id].db = &db;
      tasks[thread_id].tbl = tbl;
      tasks[thread_id].hash_idx = hash_idx;
      tasks[thread_id].btree_idx = btree_idx;
#if defined (RECOVERY)
      tasks[thread_id].thread = &thread_info;
      tasks[thread_id].txn_count = tx_count;
#endif
    }

    if (kUseContendedSet) zipf_theta = 0.;

    for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
      auto req_counts = reinterpret_cast<uint16_t*>(
          alloc.malloc_contiguous(sizeof(uint16_t) * tx_count));
      auto read_only_tx = reinterpret_cast<uint8_t*>(
          alloc.malloc_contiguous(sizeof(uint8_t) * tx_count));
      uint32_t* scan_lens = nullptr;
      if (kUseScan)
        scan_lens = reinterpret_cast<uint32_t*>(
            alloc.malloc_contiguous(sizeof(uint32_t) * tx_count));
      auto row_ids = reinterpret_cast<uint64_t*>(alloc.malloc_contiguous(
          sizeof(uint64_t) * tx_count * reqs_per_tx));
      auto column_ids = reinterpret_cast<uint8_t*>(alloc.malloc_contiguous(
          sizeof(uint8_t) * tx_count * reqs_per_tx));
      auto op_types = reinterpret_cast<uint8_t*>(alloc.malloc_contiguous(
          sizeof(uint8_t) * tx_count * reqs_per_tx));
      assert(req_counts);
      assert(row_ids);
      assert(column_ids);
      assert(op_types);
      tasks[thread_id].num_rows = num_rows;
      tasks[thread_id].tx_count = tx_count;
      tasks[thread_id].req_counts = req_counts;
      tasks[thread_id].read_only_tx = read_only_tx;
      tasks[thread_id].scan_lens = scan_lens;
      tasks[thread_id].row_ids = row_ids;
      tasks[thread_id].column_ids = column_ids;
      tasks[thread_id].op_types = op_types;

      if (kVerify) {
        auto commit_tx_i = reinterpret_cast<uint64_t*>(
            alloc.malloc_contiguous(sizeof(uint64_t) * tx_count));
        auto commit_ts = reinterpret_cast<Timestamp*>(
            alloc.malloc_contiguous(sizeof(Timestamp) * tx_count));
        auto read_ts = reinterpret_cast<Timestamp*>(alloc.malloc_contiguous(
            sizeof(Timestamp) * tx_count * reqs_per_tx));
        auto write_ts = reinterpret_cast<Timestamp*>(alloc.malloc_contiguous(
            sizeof(Timestamp) * tx_count * reqs_per_tx));
        assert(commit_tx_i);
        assert(commit_ts);
        assert(read_ts);
        ::mica::util::memset(commit_tx_i, 0, sizeof(uint64_t) * tx_count);
        ::mica::util::memset(commit_ts, 0, sizeof(Timestamp) * tx_count);
        ::mica::util::memset(read_ts, 0,
                             sizeof(Timestamp) * tx_count * reqs_per_tx);
        ::mica::util::memset(write_ts, 0,
                             sizeof(Timestamp) * tx_count * reqs_per_tx);
        tasks[thread_id].commit_tx_i = commit_tx_i;
        tasks[thread_id].commit_ts = commit_ts;
        tasks[thread_id].read_ts = read_ts;
        tasks[thread_id].write_ts = write_ts;
      }
    }

    std::vector<std::thread> threads;
    for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
      threads.emplace_back([&, thread_id] {
        ::mica::util::lcore.pin_thread(thread_id);

        uint64_t seed = 4 * thread_id * ::mica::util::rdtsc();
        uint64_t seed_mask = (uint64_t(1) << 48) - 1;
        ::mica::util::ZipfGen zg(num_rows, zipf_theta, seed & seed_mask);
        ::mica::util::Rand op_type_rand((seed + 1) & seed_mask);
        ::mica::util::Rand column_id_rand((seed + 2) & seed_mask);
        ::mica::util::Rand scan_len_rand((seed + 3) & seed_mask);
        uint32_t read_threshold =
            (uint32_t)(read_ratio * (double)((uint32_t)-1));
        uint32_t rmw_threshold =
            (uint32_t)(kReadModifyWriteRatio * (double)((uint32_t)-1));

        uint64_t req_offset = 0;

        for (uint64_t tx_i = 0; tx_i < tx_count; tx_i++) {
          bool read_only_tx = true;

          for (uint64_t req_i = 0; req_i < reqs_per_tx; req_i++) {
            size_t row_id;
            while (true) {
              if (kUseContendedSet) {
                if (req_i < kContendedReqPerTX)
                  row_id =
                      (zg.next() * 0x9ddfea08eb382d69ULL) % kContendedSetSize;
                else
                  row_id = (zg.next() * 0x9ddfea08eb382d69ULL) %
                               (num_rows - kContendedSetSize) +
                           kContendedSetSize;
              } else {
                row_id = (zg.next() * 0x9ddfea08eb382d69ULL) % num_rows;
              }
              // Avoid duplicate row IDs in a single transaction.
              uint64_t req_j;
              for (req_j = 0; req_j < req_i; req_j++)
                if (tasks[thread_id].row_ids[req_offset + req_j] == row_id)
                  break;
              if (req_j == req_i) break;
            }
            uint8_t column_id = static_cast<uint8_t>(column_id_rand.next_u32() %
                                                     (kDataSize / kColumnSize));
            tasks[thread_id].row_ids[req_offset + req_i] = row_id;
            tasks[thread_id].column_ids[req_offset + req_i] = column_id;

            if (op_type_rand.next_u32() <= read_threshold)
              tasks[thread_id].op_types[req_offset + req_i] = 0;
            else {
              tasks[thread_id].op_types[req_offset + req_i] =
                  op_type_rand.next_u32() <= rmw_threshold ? 1 : 2;
              read_only_tx = false;
            }
          }
          tasks[thread_id].req_counts[tx_i] =
              static_cast<uint16_t>(reqs_per_tx);
          tasks[thread_id].read_only_tx[tx_i] =
              static_cast<uint8_t>(read_only_tx);
          if (kUseScan) {
            tasks[thread_id].scan_lens[tx_i] =
                scan_len_rand.next_u32() % (kMaxScanLen - 1) + 1;
          }
          req_offset += reqs_per_tx;
        }
      });
    }

    while (threads.size() > 0) {
      threads.back().join();
      threads.pop_back();
    }

    for (uint64_t i=0; i< num_threads; i++) {
      printf ("task[%lu].thread_id=%lu\n", i, tasks[i].thread_id);
    }
  }

  // tbl->print_table_status();
  // if (kShowPoolStats) db.print_pool_status();

  logger.enable_logging ();
  db.reset_persist_analysis ();

  // For verification.
  std::vector<Timestamp> table_ts;

#if not defined (RECOVERY)
  for (auto phase = 0; phase < 2; phase++) {
    // if (kVerify && phase == 0) {
    //   printf("skipping warming up\n");
    //   continue;
    // }
    if (kVerify && phase == 1) {
      for (uint64_t row_id = 0; row_id < num_rows; row_id++) {
        auto rv = tbl->latest_rv(0, row_id);
        table_ts.push_back(rv->wts);
      }
    }

    if (phase == 0)
      printf("warming up\n");
    else {
      db.reset_stats();
      printf("executing workload\n");
    }

    running_threads = 0;
    stopping = 0;

    ::mica::util::memory_barrier();

    std::vector<std::thread> threads;
    for (uint64_t thread_id = 1; thread_id < num_threads; thread_id++)
      threads.emplace_back(worker_proc, &tasks[thread_id]);

    if (phase != 0 && kRunPerf) {
      int r = system("perf record -a sleep 1 &");
      // int r = system("perf record -a -g sleep 1 &");
      (void)r;
    }

    worker_proc(&tasks[0]);

    while (threads.size() > 0) {
      threads.back().join();
      threads.pop_back();
    }
  }
#else
  std::vector<std::thread> threads;
  for (uint64_t thread_id = 1; thread_id < num_threads; thread_id++) {
    threads.emplace_back(worker_proc, &tasks[thread_id]);
  }

  worker_proc(&tasks[0]);

  while (threads.size() > 0) {
    threads.back().join();
    threads.pop_back();
  }
#endif

  printf("\n");

  {
    double diff;
    {
      double min_start = 0.;
      double max_end = 0.;
      for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        double start = (double)tasks[thread_id].tv_start.tv_sec * 1. +
                       (double)tasks[thread_id].tv_start.tv_usec * 0.000001;
        double end = (double)tasks[thread_id].tv_end.tv_sec * 1. +
                     (double)tasks[thread_id].tv_end.tv_usec * 0.000001;
        if (thread_id == 0 || min_start > start) min_start = start;
        if (thread_id == 0 || max_end < end) max_end = end;
      }

      diff = max_end - min_start;
    }
    double total_time = diff * static_cast<double>(num_threads);

    uint64_t total_committed = 0;
    uint64_t total_scanned = 0;
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
      total_committed += tasks[thread_id].committed;
      if (kUseScan) total_scanned += tasks[thread_id].scanned;
    }
    printf("throughput:                   %7.3lf M/sec\n",
           static_cast<double>(total_committed) / diff * 0.000001);

    printf("page_used:                    %lu pages\n", db.page_pool()->both_page_num());
    printf("memory_size:                  %7.3lf GB\n", db.page_pool()->both_memory_use());

    printf("aep_page_used:                %lu pages\n", db.aep_page_pool()->both_page_num());
    printf("aep_memory_size:              %7.3lf GB\n", db.aep_page_pool()->both_memory_use());

    printf("aep_clwb_cnt:                 %7.3lf M\n",  db.get_clwb_cnt ());
    printf("aep_sfence_cnt:               %7.3lf M\n",  db.get_sfence_cnt ());
    printf("aep_persist_size:             %7.3lf GB\n", db.get_persist_size());

    printf ("\nDRAM Page Pool:\n");
    db.page_pool()->print_status();
    printf ("NVRAM Page Pool:\n");
    db.aep_page_pool()->print_status();
    printf ("\n");

    if (kUseScan)
      printf("scan throughput:              %7.3lf M/sec\n",
             static_cast<double>(total_scanned) / diff * 0.000001);

    db.print_stats(diff, total_time);

    printf ("Table Status:\n");
    tbl->print_table_status();

    if (hash_idx != nullptr) {
      printf ("HashIdx Status:\n");
      hash_idx->index_table()->print_table_status();
    }
    if (btree_idx != nullptr) {
      printf ("BtreeIdx Status:\n");
      btree_idx->index_table()->print_table_status();
    }

    db.print_pool_status();
    logger.log_print_status ();
  }

#if defined (CZEN) && defined (DEBUG)
  {
        // check the table after transactions is initialized as think
        {
          db.activate(static_cast<uint16_t>(0));
          printf ("check initilize process after transactions\n");
          for (uint64_t i=0; i< num_rows; i++) {
            PmemTransaction tx(db.context(static_cast<uint16_t>(0)));
            tx.aep_begin();
            uint64_t vrow_id = 0;
            auto lookup_result =
              hash_idx->lookup(&tx, i, kSkipValidationForIndexAccess,
                               [&vrow_id](auto& k, auto& v) {
                                 (void)k;
                                 vrow_id = v;
                                 return false;
                               });
            if (lookup_result != 1 || lookup_result == HashIndex::kHaveToAbort) {
              assert(false);
            }

            // printf ("key=%lu\tval=%lu\n", i, vrow_id); 
 
            PmemRowAccessHandle rah(&tx);

            if (!rah.aep_peek_row(tbl, 0, vrow_id, false, true, false) ||
                !rah.aep_read_row()) {

              // printf ("abort because of peek read lookup! key=%lu\tval=%lu\n", i, vrow_id);
              tx.aep_abort();
              break;
            }

            printf ("key=%lu\tval=%lu\tval=%p\trah.row_id()=%lu\thead->primkey=%lu\tread_rv=%p\twrite_rv=%p\taep_rv=%p\taep_rv->pkey=%lu\n",
                     i, 
                     vrow_id,
                     reinterpret_cast<void*>(vrow_id),
                     rah.row_id(),
                     tbl->head(0, rah.row_id())->pkey,
                     rah.row_access_item()->read_rv,
                     rah.row_access_item()->write_rv,
                     rah.row_access_item()->write_rv?
                     rah.row_access_item()->write_rv->aep_rv:
                     rah.row_access_item()->read_rv->aep_rv,
                     rah.row_access_item()->write_rv?
                     rah.row_access_item()->write_rv->aep_rv->pkey:
                     rah.row_access_item()->read_rv->aep_rv->pkey);

            tx.aep_commit();
          }
          db.deactivate(0);
        }
  }
#endif

  if (kVerify) {
    printf("verifying\n");
    const bool print_verification = false;
    // const bool print_verification = true;

    if (kUseSnapshot)
      printf(
          "warning: verification currently does not support transactions using "
          "snapshots\n");

    uint64_t total_tx_i = 0;
    uint64_t current_commit_i[num_threads];
    for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++)
      current_commit_i[thread_id] = 0;

    while (true) {
      // Simple but slow way to find the next executed transaction.
      uint64_t next_thread_id = static_cast<uint64_t>(-1);
      Timestamp next_ts = Timestamp::make(0, 0, 0);
      for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        Task* task = &tasks[thread_id];

        if (current_commit_i[thread_id] >= task->committed) continue;

        uint64_t tx_i = task->commit_tx_i[current_commit_i[thread_id]];
        if (next_ts > task->commit_ts[tx_i] ||
            next_thread_id == static_cast<uint64_t>(-1)) {
          next_thread_id = thread_id;
          next_ts = task->commit_ts[tx_i];
        }
      }
      if (next_thread_id == static_cast<uint64_t>(-1)) break;

      Task* task = &tasks[next_thread_id];
      uint64_t tx_i = task->commit_tx_i[current_commit_i[next_thread_id]];
      // XXX: Assume the constant number of reqs per tx.
      uint64_t total_req_i = reqs_per_tx * tx_i;

      if (print_verification)
        printf("thread=%2" PRIu64 " tx=%" PRIu64 " commit_ts=0x%" PRIx64 "\n",
               next_thread_id, tx_i, next_ts.t2);

      for (uint64_t req_j = 0; req_j < task->req_counts[tx_i]; req_j++) {
        const Timestamp& read_ts = task->read_ts[total_req_i + req_j];
        const Timestamp& write_ts = task->write_ts[total_req_i + req_j];
        uint64_t row_id = task->row_ids[total_req_i + req_j];
        bool is_read = task->op_types[total_req_i + req_j] == 0;
        bool is_rmw = task->op_types[total_req_i + req_j] == 1;

        if (print_verification)
          printf("  req %2" PRIu64 " row=%9" PRIu64
                 " is_read=%d read_ts=0x%" PRIx64 "\n",
                 req_j, row_id, is_read ? 1 : 0, read_ts.t2);

        // The read set timestamp must be smaller than the commit timestamp
        // (consistency).
        if (read_ts >= next_ts) {
          printf("verification failed at %" PRIu64
                 " (read ts must predate commit ts)\n",
                 total_tx_i);
          assert(false);
          return 1;
        }

        if (is_read || is_rmw) {
          // There must be no extra version between the read and commit
          // timestamp unless it is a write-only operation (atomicity).
          const Timestamp& stored_ts = table_ts[row_id];
          if (stored_ts != read_ts) {
            printf("verification failed at %" PRIu64
                   " (read ts mismatch; expected 0x%" PRIx64 ", got 0x%" PRIx64
                   ")\n",
                   total_tx_i, read_ts.t2, stored_ts.t2);
            assert(false);
            return 1;
          }
        }

        // Register the new version.  It is guranteed to be the latest version
        // because we chose the transaction in their commit ts order.
        // if (!is_read) table_ts[row_id] = next_ts;
        // Since we now have a promotion, a read may be actually a write.
        // It keeps the old timestamp or updated to the transaction's timestamp.
        assert(write_ts == read_ts || write_ts == next_ts);
        table_ts[row_id] = write_ts;
      }

      current_commit_i[next_thread_id]++;
      total_tx_i++;
    }

    // Final check.
    for (uint64_t row_id = 0; row_id < num_rows; row_id++) {
      auto stored_ts = table_ts[row_id];
      auto rv = tbl->latest_rv(0, row_id);
      auto read_ts = rv->wts;

      if (stored_ts != read_ts) {
        printf("verification failed at %" PRIu64
               " (read ts mismatch; expected 0x%" PRIx64 ", got 0x%" PRIx64
               ")\n",
               total_tx_i, read_ts.t2, stored_ts.t2);
        assert(false);
        return 1;
      }
    }

    printf("passed verification\n");
    printf("\n");
  }

  {
    printf("cleaning up\n");
    for (uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
      alloc.free_contiguous(tasks[thread_id].req_counts);
      alloc.free_contiguous(tasks[thread_id].read_only_tx);
      if (kUseScan) alloc.free_contiguous(tasks[thread_id].scan_lens);
      alloc.free_contiguous(tasks[thread_id].row_ids);
      alloc.free_contiguous(tasks[thread_id].column_ids);
      alloc.free_contiguous(tasks[thread_id].op_types);
      if (kVerify) {
        alloc.free_contiguous(tasks[thread_id].commit_tx_i);
        alloc.free_contiguous(tasks[thread_id].commit_ts);
        alloc.free_contiguous(tasks[thread_id].read_ts);
        alloc.free_contiguous(tasks[thread_id].write_ts);
      }
    }
  }

  return EXIT_SUCCESS;
}

#if defined (RECOVERY)
int main(int argc, const char* argv[]) {
  // for recovery, require data initilized as 0 before test
  int ret = system ("rm /mnt/mypmem0/liugang/mypmem*");
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
