#pragma once

#include <stdint.h>
#include <map>
#include <string>
#include <fstream>

#include "mica/transaction/db.h"

#define MICA_ALLOC_SIZE    (10UL<<30)
#define MICA_AEPALLOC_SIZE (40UL<<30)

class Query_queue;
class workload;

struct DBConfig : public ::mica::transaction::BasicDBConfig {
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
typedef MICADB::HashIndexNonuniqueU64 MICAIndex;
typedef MICADB::BTreeIndexUniqueU64 MICAOrderedIndex;
typedef ::mica::transaction::RowVersion<DBConfig> MICARowVersion;
typedef ::mica::transaction::RowAccessHandle<DBConfig> MICARowAccessHandle;
typedef ::mica::transaction::PmemRowAccessHandle<DBConfig> MICAPmemRowAccessHandle;
typedef ::mica::transaction::RowAccessHandlePeekOnly<DBConfig> MICARowAccessHandlePeekOnly;
typedef ::mica::transaction::PmemRowAccessHandlePeekOnly<DBConfig> MICAPmemRowAccessHandlePeekOnly;
typedef ::mica::transaction::Transaction<DBConfig> MICATransaction;
typedef ::mica::transaction::PmemTransaction<DBConfig> MICAPmemTransaction;
typedef ::mica::transaction::Result MICAResult;

typedef uint64_t ts_t;      // time stamp type
typedef uint64_t idx_key_t; // key id for index

typedef MICAIndex        HASH_INDEX;
typedef MICAOrderedIndex ORDERED_INDEX;

/* general concurrency control */
enum RC { RCOK, Commit, Abort, WAIT, ERROR, FINISH};
enum access_t {RD, WR, XP, SCAN, PEEK, SKIP};

Query_queue * query_queue;
bool volatile warmup_finish = false;
pthread_barrier_t warmup_bar;
pthread_barrier_t start_bar;

uint64_t g_transaction_cnt = 200000;
uint64_t g_thread_cnt = 16;
uint64_t g_num_wh = 128;
uint64_t g_part_cnt = 1;

double   g_perc_payment = 0.5;
bool     g_wh_update = true;
char    *output_file = NULL;
uint32_t g_max_items = 100000;
uint32_t g_cust_per_dist = 3000;
uint64_t g_max_orderline = uint64_t(1) << 32;

#define DIST_PER_WARE (10)
#define PAGE_SIZE (1UL<<21)
#define CL_SIZE (64)
#define WARMUP  (66666)
using namespace std;

