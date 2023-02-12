#pragma once
#ifndef MICA_TRANSACTION_LOGGING_H_
#define MICA_TRANSACTION_LOGGING_H_

#include "mica/common.h"
#include "mica/transaction/db.h"
#include "mica/transaction/row.h"
#include "mica/transaction/row_version_pool.h"
#include "mica/transaction/context.h"
#include "mica/transaction/transaction.h"

namespace mica {
namespace transaction {

template <class StaticConfig>
class LoggerInterface {
 public:
  virtual bool log(const Transaction<StaticConfig>* tx) {
    (void) tx;
    // printf ("Not support! LoggerInterface log!\n");
    return true;
  }
  LoggerInterface (): logging_enabled_(true) {}

  virtual void enable_logging  () { logging_enabled_ = true; }
  virtual void disable_logging () { logging_enabled_ = false; }
  virtual void set_logging_thread_count (uint16_t thread_cnt) {
    logging_thread_count_ = thread_cnt;
  }

  virtual bool log_commit_state (const Transaction<StaticConfig> *tx) {
    (void) tx;
    // printf ("Not support! LoggerInterface log_commit_state!\n");
    return true;
  }

  virtual void log_print_status () {}
 protected:
  bool logging_enabled_;
  uint16_t logging_thread_count_;
};

template <class StaticConfig>
class NullLogger : public LoggerInterface<StaticConfig> {
 public:
  bool log(const Transaction<StaticConfig>* tx) {
    (void)tx;
    return true;
  }

  void log_print_status () {
    printf ("total log: 0 MB\n");
  }
};

template <class StaticConfig>
class PmemLogger : public LoggerInterface<StaticConfig> {
 private:
  PagePool<StaticConfig> *aep_page_pool_;
  struct LogPage {
    std::vector<char*> pages;
    char *p;
    char *pstart,*pend;

    LogPage () {
      p      = nullptr;
      pstart = nullptr;
      pend   = nullptr;
    }
  };
  LogPage log_partition_by_thread_[StaticConfig::kMaxLCoreCount];

  struct LogNum {
    uint64_t log_cnt;
    uint64_t log_size;

    LogNum () {
      log_cnt = 0UL;
      log_size= 0UL;
    }
  };
  struct LogNum log_num_by_thread_[StaticConfig::kMaxLCoreCount];

  #if defined (CWBL)
  uint64_t wbl_log_count_[StaticConfig::kMaxLCoreCount];
  uint64_t wbl_log_goup_count_[StaticConfig::kMaxLCoreCount];
  #endif

  // for recovery all type of method
  friend class DB<StaticConfig>;

 public:
  PmemLogger (PagePool<StaticConfig> *aep_page_pool) 
    : LoggerInterface<StaticConfig>(), aep_page_pool_(aep_page_pool) {
    #if defined (CWBL)
    for (uint64_t i=0; i<StaticConfig::kMaxLCoreCount; i++) {
      wbl_log_count_[i] = 0;
      wbl_log_goup_count_[i] = 0;
    }
    #endif
  }

  ~PmemLogger () {
    for (uint16_t i=0; i< LoggerInterface<StaticConfig>::logging_thread_count_; i++) {
      LogPage &lp = log_partition_by_thread_[i];
      for (size_t j=0; j< lp.pages.size(); j++) {
        aep_page_pool_->free (lp.pages[j]);
      }
    }
  }

  void log_print_status () {
    uint64_t total_log_cnt  = 0UL;
    uint64_t total_log_size = 0UL;

    printf ("log on aep status:\n");
    for (uint16_t i=0; i< LoggerInterface<StaticConfig>::logging_thread_count_; i++) {

      auto cnt  = log_num_by_thread_[i].log_cnt;
      auto size = log_num_by_thread_[i].log_size;
      auto size_d = static_cast<double>(size)/1000000.0;

      total_log_cnt  += cnt;
      total_log_size += size;

      printf ("  thread=%u log_num=%7lu log_size=%7.3lf MB\n", i, cnt, size_d);
    }

    auto t_size_d = static_cast<double>(total_log_size)/1000000.0;
    printf ("log total: log_num=%7lu log_size=%7.3lf MB\n", total_log_cnt, t_size_d);
  }

  bool log (const Transaction<StaticConfig> *tx) {
    if (LoggerInterface<StaticConfig>::logging_enabled_ == false) {
      return true;
    }

    Context<StaticConfig> *ctx = const_cast<Context<StaticConfig>*>(tx->context());
    size_t thread_id = tx->context()->thread_id();

#if defined (CWBL)
    wbl_log_count_[thread_id] ++;
    if (wbl_log_goup_count_[thread_id]+StaticConfig::kWblCommitPerInterval 
        < wbl_log_count_[thread_id]) {
      return true;
    }
    else {
      wbl_log_goup_count_[thread_id] += StaticConfig::kWblCommitPerInterval;
    }
#endif

    while (1) {
      char *bf_s = log_partition_by_thread_[thread_id].p;
      char *bf_e = log_partition_by_thread_[thread_id].pend;
      if (bf_s == nullptr || bf_s+sizeof(uint64_t)+sizeof(size_t) > bf_e) {
        get_an_aep_page (thread_id);
        continue;
      }
      size_t sz = tx->construct_log (bf_s, bf_e);  // first log entry then write ts
      if (sz != 0) {

        ctx->context_pmem_flush (bf_s+sizeof(uint64_t), sz-sizeof(uint64_t));

#if defined (test)
        // for write ahead log need to write committed log
        *(reinterpret_cast<uint64_t*>(bf_s)) = 0;
        (const_cast<Transaction<StaticConfig>*>(tx))->construct_pre_commit_log (bf_s);
#elif defined (CWBL)
        // write behind log only write the important t2
        *(reinterpret_cast<uint64_t*>(bf_s)) = tx->ts().t2;
#elif defined (CZEN)
        // not important, not called
        *(reinterpret_cast<uint64_t*>(bf_s)) = tx->ts().t2;
#endif

        ctx->context_pmem_persist (bf_s, sizeof(uint64_t));

        log_partition_by_thread_[thread_id].p += sz;

#if defined (DEBUG)
        // DEBUG
        printf ("logcommit t2=%lu sz=%lu\n", tx->ts().t2, sz);
        char *p = bf_s+sizeof(uint64_t)+sizeof(size_t);
        while (p<bf_s+sz) {
          void* tbl = reinterpret_cast<void*>(p);
          uint16_t cf = *reinterpret_cast<uint16_t*>(p+8);
          uint64_t row = *reinterpret_cast<uint64_t*>(p+10);
          uint64_t pkey = *reinterpret_cast<uint64_t*>(p+18);
          uint8_t intention = *reinterpret_cast<uint8_t*>(p+26);
          uint64_t dtsz = *reinterpret_cast<uint64_t*>(p+27);
          printf ("  bf_s=%p cf=%u row=%lu pkey=%lu intention=%u dtsz=%lu\n",
            bf_s,cf,row,pkey,intention,dtsz);
          p += dtsz+35;
        }
#endif

        log_num_by_thread_[thread_id].log_cnt  += 1UL;
        log_num_by_thread_[thread_id].log_size += sz;

        break;
      }
      else {
        // *(reinterpret_cast<uint64_t*>(bf_s)) = 0;
        get_an_aep_page (thread_id); 
      } 
    }
    return true;
  }

  bool log_commit_state (const Transaction<StaticConfig> *tx) {
    if (LoggerInterface<StaticConfig>::logging_enabled_ == false) {
      return true;
    }

#if defined (DEBUG)
    printf ("t2=%lu commit\n", tx->ts().t2);
#endif

    size_t thread_id = tx->context()->thread_id();
    log_num_by_thread_[thread_id].log_cnt  += 1UL;
    log_num_by_thread_[thread_id].log_size += sizeof(uint64_t);
   
    return tx->construct_commit_log ();
  }

 private:
  bool get_an_aep_page (size_t thread_id) {
    char *page = aep_page_pool_->allocate();
    if (page == nullptr) {
      return false;
    }
    LogPage &lp = log_partition_by_thread_[thread_id];
    lp.pages.push_back (page);
    lp.p = lp.pstart = page;
    lp.pend = lp.p + PagePool<StaticConfig>::kPageSize;
    return true; 
  }  
};

template <class StaticConfig>
class PmemCheckpoint {
 private:
  PagePool<StaticConfig> *aep_page_pool_;

 public:
  uint16_t cf_count_;
  size_t dt_size_[StaticConfig::kMaxColumnFamilyCount];
  size_t cf_size_[StaticConfig::kMaxColumnFamilyCount];
  struct CheckpointPage {
    std::vector<char*> pages;
    char *p;
    char *pstart,*pend;
    // iterator
    size_t it_pg;
    char   *it_p,*it_pend;
    CheckpointPage() {
      p      = nullptr;
      pstart = nullptr;
      pend   = nullptr;
      it_pg  = 0;
      it_p   = nullptr;
      it_pend= nullptr;
    }
  };
  CheckpointPage checkpoint_partition_by_column_family_[StaticConfig::kMaxColumnFamilyCount];
  uint64_t row_num_;

  struct CheckRow {
    uint64_t pkey;
    char data[0] __attribute__((aligned(8)));
  };

 public:
  PmemCheckpoint (PagePool<StaticConfig> *aep_page_pool, 
                 uint16_t cf_count, 
                 const uint64_t *data_size_hint) 
    : aep_page_pool_(aep_page_pool), cf_count_(cf_count) {
    for (uint16_t i=0; i<cf_count_; i++) {
      dt_size_[i] = data_size_hint[i];
      cf_size_[i] = dt_size_[i]+sizeof(CheckRow);
    }
    row_num_ = 0;
  }
  ~PmemCheckpoint () {
    for (uint16_t i=0; i< StaticConfig::kMaxColumnFamilyCount; i++) {
      CheckpointPage &cp = checkpoint_partition_by_column_family_[i];
      for (size_t j=0; j< cp.pages.size(); j++) {
        aep_page_pool_->free (cp.pages[j]);
      }
    }
  }

  void checkpoint_print_status () {
    uint64_t page_num = 0UL;

    for (uint16_t i=0; i< cf_count_; i++) {
      page_num += checkpoint_partition_by_column_family_[i].pages.size();
    }
    uint64_t page_size = page_num*aep_page_pool_->kPageSize;
    double page_size_d = static_cast<double>(page_size);

    printf ("checkpoint_row:             %lu\n", row_num_);
    printf ("checkpoint_page_num:        %lu\n", page_num);
    printf ("checkpoint_page_size:       %7.3lf\n", page_size_d);
  }

  uint64_t check_column_family (char *p, uint16_t cf_id, uint64_t hdk) {
    CheckpointPage &cp = checkpoint_partition_by_column_family_[cf_id];
    if (cp.p == nullptr || cp.p+cf_size_[cf_id] > cp.pend) {
      if (get_an_aep_page(cf_id) == false) {
        printf ("Error: checkpoint no page!\n");
        return static_cast<uint64_t>(-1);
      }
    }
    CheckRow *crp = reinterpret_cast<CheckRow*>(cp.p);
    crp->pkey = hdk;
    ::mica::util::memcpy (crp->data, p, dt_size_[cf_id]);

    pmem_persist (cp.p, cf_size_[cf_id]);
    cp.p += cf_size_[cf_id];

    return cf_size_[cf_id];
  }

 private:
  bool get_an_aep_page (uint16_t cf_id) {
    char *page = aep_page_pool_->allocate();
    if (page == nullptr) {
      return false;
    }
    CheckpointPage &lp = checkpoint_partition_by_column_family_[cf_id];
    lp.pages.push_back (page);
    lp.p = lp.pstart = page;
    lp.pend = lp.p + PagePool<StaticConfig>::kPageSize;
    return true; 
  }

  friend class DB<StaticConfig>;
};

}
}

#endif
