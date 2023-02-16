#pragma once

#include "stdint.h"
#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <typeinfo>
#include <list>
#include <mm_malloc.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "pthread.h"
#include "config.h"
#include "stats.h"
#include "dl_detect.h"
#ifndef NOGRAPHITE
#include "carbon_user.h"
#endif

#if CC_ALG == MICA

#define NDEBUG
#include "mica/transaction/db.h"
#undef NDEBUG
struct DBConfig : public ::mica::transaction::BasicDBConfig {
  // static constexpr bool kVerbose = true;

#if MICA_NO_PRE_VALIDATION
  static constexpr bool kPreValidation = false;
#endif
#if MICA_NO_INSERT_NEWEST_VERSION_ONLY
  static constexpr bool kInsertNewestVersionOnly = false;
#endif
#if MICA_NO_SORT_WRITE_SET_BY_CONTENTION
  static constexpr bool kSortWriteSetByContention = false;
#endif
#if MICA_NO_STRAGGLER_AVOIDANCE
  static constexpr bool kStragglerAvoidance = false;
#endif
#if MICA_NO_WAIT_FOR_PENDING
  static constexpr bool kNoWaitForPending = true;
#endif
#if MICA_NO_INLINING
  static constexpr bool kInlinedRowVersion = false;
#endif

#if MICA_NO_BACKOFF
    static constexpr bool kBackoff = false;
#endif

  // static constexpr bool kPrintBackoff = true;
  // static constexpr bool kPairwiseSleeping = true;

#if MICA_USE_FIXED_BACKOFF
  static constexpr double kBackoffMin = MICA_FIXED_BACKOFF;
  static constexpr double kBackoffMax = MICA_FIXED_BACKOFF;
#endif

#if MICA_USE_SLOW_GC
  static constexpr int64_t kMinQuiescenceInterval = MICA_SLOW_GC;
#endif

  // typedef ::mica::transaction::WideTimestamp Timestamp;
  // typedef ::mica::transaction::WideConcurrentTimestamp ConcurrentTimestamp;
#if MICA_NO_TSC
  typedef ::mica::transaction::CentralizedTimestamp Timestamp;
  typedef ::mica::transaction::CentralizedConcurrentTimestamp ConcurrentTimestamp;
#endif

  // static constexpr bool kCollectCommitStats = false;
  // static constexpr bool kCollectExtraCommitStats = true;
  // static constexpr bool kCollectProcessingStats = true;
  // static constexpr bool kCollectROTXStalenessStats = true;
  // typedef ::mica::transaction::ActiveTiming Timing;

  typedef ::mica::transaction::LoggerInterface<DBConfig> LoggerInterface;
  typedef ::mica::transaction::NullLogger<DBConfig> Logger;
  typedef ::mica::transaction::PmemLogger<DBConfig> PmemLogger;
 
};

typedef DBConfig::Alloc MICAAlloc;
typedef DBConfig::PmemAlloc MICAPmemAlloc;

typedef DBConfig::LoggerInterface MICALoggerInterface;
typedef DBConfig::Logger MICALogger;
typedef DBConfig::PmemLogger MICAPmemLogger;

typedef DBConfig::Timing MICATiming;
typedef ::mica::transaction::PagePool<DBConfig> MICAPagePool;
typedef ::mica::transaction::DB<DBConfig> MICADB;
typedef ::mica::transaction::Table<DBConfig> MICATable;

// debug
// typedef MICADB::HashIndexNonuniqueU64 MICAIndex;
typedef MICADB::HashIndexNonuniqueU64 MICAIndex;

// typedef MICADB::BTreeIndexNonuniqueU64 MICAOrderedIndex;
typedef MICADB::BTreeIndexUniqueU64 MICAOrderedIndex;
typedef ::mica::transaction::RowVersion<DBConfig> MICARowVersion;

typedef ::mica::transaction::RowAccessHandle<DBConfig> MICARowAccessHandle;
typedef ::mica::transaction::PmemRowAccessHandle<DBConfig> MICAPmemRowAccessHandle;
typedef ::mica::transaction::RowAccessHandlePeekOnly<DBConfig> MICARowAccessHandlePeekOnly;
typedef ::mica::transaction::PmemRowAccessHandlePeekOnly<DBConfig> MICAPmemRowAccessHandlePeekOnly;

typedef ::mica::transaction::Transaction<DBConfig> MICATransaction;
typedef ::mica::transaction::PmemTransaction<DBConfig> MICAPmemTransaction;
typedef ::mica::transaction::Result MICAResult;

#if defined (RECOVERY)
template <class StaticConfig>
class ThreadInfoTemplate {
 public:
  std::mutex mtx;
  pthread_t first;
  uint16_t thread_cnt;
  uint16_t active_cnt;
  #if CC_ALG == MICA
  MICADB *db;
  #endif
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
#endif

#else
#include "mica/util/stopwatch.h"
#include "mica/util/latency.h"
#endif

using namespace std;

class mem_alloc;
class Stats;
class DL_detect;
class Manager;
class Query_queue;
class Plock;
class OptCC;
class VLLMan;

typedef uint8_t UInt8;
typedef int8_t SInt8;
typedef uint16_t UInt16;
typedef int16_t SInt16;
typedef uint32_t UInt32;
typedef int32_t SInt32;
typedef uint64_t UInt64;
typedef int64_t SInt64;

typedef uint64_t ts_t; // time stamp type

/******************************************/
// Global Data Structure
/******************************************/
extern mem_alloc mem_allocator;
extern Stats stats;
extern DL_detect dl_detector;
extern Manager * glob_manager;
extern Query_queue * query_queue;
extern Plock part_lock_man;
extern OptCC occ_man;
#if CC_ALG == VLL
extern VLLMan vll_man;
#endif

extern bool volatile warmup_finish;
extern bool volatile enable_thread_mem_pool;
extern pthread_barrier_t warmup_bar;
extern pthread_barrier_t start_bar;
#ifndef NOGRAPHITE
extern carbon_barrier_t enable_barrier;
#endif

/******************************************/
// Global Parameter
/******************************************/
extern bool g_part_alloc;
extern bool g_mem_pad;
extern bool g_prt_lat_distr;
extern UInt32 g_part_cnt;
extern UInt32 g_virtual_part_cnt;
extern UInt32 g_thread_cnt;
extern ts_t g_abort_penalty;
extern bool g_central_man;
extern UInt32 g_ts_alloc;
extern bool g_key_order;
extern bool g_no_dl;
extern ts_t g_timeout;
extern ts_t g_dl_loop_detect;
extern bool g_ts_batch_alloc;
extern UInt32 g_ts_batch_num;

extern map<string, string> g_params;

// YCSB
extern UInt32 g_cc_alg;
extern ts_t g_query_intvl;
extern UInt32 g_part_per_txn;
extern double g_perc_multi_part;
extern double g_read_perc;
extern double g_write_perc;
extern double g_zipf_theta;
extern UInt64 g_synth_table_size;
extern UInt32 g_req_per_query;
extern UInt32 g_field_per_tuple;
extern UInt32 g_init_parallelism;

// TPCC
extern UInt32 g_num_wh;
extern double g_perc_payment;
extern bool g_wh_update;
extern char * output_file;
extern UInt32 g_max_items;
extern UInt32 g_cust_per_dist;
extern uint64_t g_max_orderline;

// TATP
extern uint64_t g_sub_size;

enum RC { RCOK, Commit, Abort, WAIT, ERROR, FINISH};

/* Thread */
typedef uint64_t txnid_t;

/* Txn */
typedef uint64_t txn_t;

/* Table and Row */
typedef uint64_t rid_t; // row id
typedef uint64_t pgid_t; // page id



/* INDEX */
enum latch_t {LATCH_EX, LATCH_SH, LATCH_NONE};
// accessing type determines the latch type on nodes
enum idx_acc_t {INDEX_INSERT, INDEX_READ, INDEX_NONE};
typedef uint64_t idx_key_t; // key id for index
typedef uint64_t (*func_ptr)(idx_key_t);	// part_id func_ptr(index_key);

/* general concurrency control */
enum access_t {RD, WR, XP, SCAN, PEEK, SKIP};
/* LOCK */
enum lock_t {LOCK_EX, LOCK_SH, LOCK_NONE };
/* TIMESTAMP */
enum TsType {R_REQ, W_REQ, P_REQ, XP_REQ};


#define MSG(str, args...) { \
	printf("[%s : %d] " str, __FILE__, __LINE__, args); } \
//	printf(args); }

// principal index structure. The workload may decide to use a different
// index structure for specific purposes. (e.g. non-primary key access should use hash)
#if (INDEX_STRUCT == IDX_MICA)

#define HASH_INDEX		IndexMICA
// #define HASH_INDEX		OrderedIndexMICA
#define ARRAY_INDEX		IndexArray
#if !SIMPLE_INDEX_UPDATE
#define ORDERED_INDEX		OrderedIndexMICA

#else
#define ORDERED_INDEX		IndexMICAMBTree
#define IDX_MICA_USE_MBTREE
#endif

#else  // IDX_HASH

#define HASH_INDEX		IndexHash
// #define ORDERED_INDEX		index_btree
#define ARRAY_INDEX		IndexArray
#define ORDERED_INDEX		IndexMBTree
#endif

/************************************************/
// constants
/************************************************/
#ifndef UINT64_MAX
#define UINT64_MAX 		18446744073709551615UL
#endif // UINT64_MAX
