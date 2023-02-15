#pragma once
#ifndef MICA_TRANSACTION_CONTEXT_H_
#define MICA_TRANSACTION_CONTEXT_H_

#include <queue>
#include "mica/transaction/stats.h"
#include "mica/transaction/row.h"
#include "mica/transaction/table.h"
#include "mica/transaction/db.h"
#include "mica/transaction/row_version_pool.h"
#include "mica/util/memcpy.h"
#include "mica/util/rand.h"
#include "mica/util/latency.h"

#include <libpmem.h>

// #include "mica/util/queue.h"

namespace mica {
namespace transaction {
template <class StaticConfig>
struct RowHead;
template <class StaticConfig>
class Table;
template <class StaticConfig>
class Transaction;
template <class StaticConfig>
class DB;

template <class StaticConfig>
class Context {
 public:
  typedef typename StaticConfig::Timestamp Timestamp;
  typedef typename StaticConfig::ConcurrentTimestamp ConcurrentTimestamp;
  typedef typename StaticConfig::Timing Timing;

  Context(DB<StaticConfig>* db, uint16_t thread_id, uint8_t numa_id)
      : db_(db),
        thread_id_(thread_id),
        numa_id_(numa_id),
        backoff_rand_(static_cast<uint64_t>(thread_id)),
        timing_stack_(&stats_, db_->sw()) {
    if (StaticConfig::kPairwiseSleeping) {
      auto active_count = db_->thread_count();
      auto count = ::mica::util::lcore.lcore_count();
      auto half_count = count / 2;
      if (thread_id_ < half_count) {
        if (thread_id_ + half_count < active_count)
          pair_selector_ = 0;
        else
          pair_selector_ = 2;  // No matching hyperthread pair.
      } else
        pair_selector_ = 1;
    }

    clock_ = 0;
    clock_boost_ = 0;
    adjusted_clock_ = 0;

    next_sync_thread_id_ = 0;

    last_tsc_ = ::mica::util::rdtsc();
    last_quiescence_ = db_->sw()->now();
    last_clock_sync_ = db_->sw()->now();

    persist_size_= 0UL;
    clwb_cnt_ = 0UL;
    sfence_cnt_ = 0UL;

    #if MEMORY_ADDRESS_WRITE
    size_t size = 8UL<<30;
    int charnum = snprintf (address_of_dirty_file_path_, 256, "/mnt/mypmem0/addressofdirty%d", thread_id_);
    printf ("addr_dirty: %s\n", address_of_dirty_file_path_);

    (void) charnum;
    assert (charnum < 256);

    void *pmemaddr = nullptr;
    size_t mapped_len = size;
    int is_pmem;
    if ((pmemaddr = pmem_map_file(address_of_dirty_file_path_, size, PMEM_FILE_CREATE, 
                                  0666, &mapped_len, &is_pmem)) == NULL) {
      perror("pmem_map_file");
      exit(1);
    }
    assert ((is_pmem) && (mapped_len == size));

    address_of_dirty_ = (void**)pmemaddr;
    address_of_dirty_off_ = 0;
    printf ("thread %u address_dirty init!\n", thread_id_);
    #endif
  }
    #if MEMORY_ADDRESS_WRITE
    void** address_of_dirty_;
    uint64_t address_of_dirty_off_;
    char address_of_dirty_file_path_[256];
    #endif
   


  ~Context() {}

  DB<StaticConfig>* db() { return db_; }

  uint64_t clock() const { return clock_; }

  Timestamp wts() const { return wts_.get(); }
  Timestamp rts() const { return rts_.get(); }

  uint16_t thread_id() const { return thread_id_; }
  uint8_t numa_id() const { return numa_id_; }

  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  const ::mica::util::Latency& inter_commit_latency() const {
    return inter_commit_latency_;
  }
  ::mica::util::Latency& inter_commit_latency() {
    return inter_commit_latency_;
  }

  const ::mica::util::Latency& commit_latency() const {
    return commit_latency_;
  }
  ::mica::util::Latency& commit_latency() { return commit_latency_; }

  const ::mica::util::Latency& abort_latency() const { return abort_latency_; }
  ::mica::util::Latency& abort_latency() { return abort_latency_; }

  const ::mica::util::Latency& ro_tx_staleness() const {
    return ro_tx_staleness_;
  }
  ::mica::util::Latency& ro_tx_staleness() { return ro_tx_staleness_; }

  TimingStack* timing_stack() { return &timing_stack_; }

  void set_clock(uint64_t ref_clock) {
    clock_ = ref_clock;
    clock_boost_ = 0;
    adjusted_clock_ = ref_clock;

    last_tsc_ = ::mica::util::rdtsc();
    last_clock_sync_ = db_->sw()->now();
  }

  void synchronize_clock() {
    uint16_t thread_id = next_sync_thread_id_;
    if (++next_sync_thread_id_ == db_->thread_count()) next_sync_thread_id_ = 0;

    if (!db_->is_active(thread_id)) return;

    auto ctx = db_->context(thread_id);

    // Update the local clock.
    update_clock();

    ::mica::util::memory_barrier();

    // Read the other thread's clock.  This clock is slightly behind because of
    // the latency.  We cannot correct this error because this is one-way
    // synchronization.
    auto clock = ctx->clock();

    // Take it if the other one is faster.
    int64_t clock_diff = static_cast<int64_t>(clock - clock_);

    if (clock_diff > 0) clock_ = clock;

    // Note that we do not reset last_clock_sync_ so that the next clock update
    // can include the latency of reading the remote clock.

    // TODO: Would it be more accurate to consider only the half of the clock
    // read latency, as if in the network I/O?
  }

  void update_clock() {
    uint64_t tsc = ::mica::util::rdtsc();

    int64_t tsc_diff = static_cast<int64_t>(tsc - last_tsc_);
    if (tsc_diff <= 0)
      tsc_diff = 1;
    else if (tsc_diff > StaticConfig::kMaxClockIncrement)
      tsc_diff = StaticConfig::kMaxClockIncrement;

    clock_ += static_cast<uint64_t>(tsc_diff);
    last_tsc_ = tsc;
  }

  Timestamp generate_timestamp(bool for_peek_only_transaction = false) {
    update_clock();

    auto adjusted_clock = clock_ + clock_boost_;
    if (static_cast<int64_t>(adjusted_clock - adjusted_clock_) <= 0)
      adjusted_clock = adjusted_clock_ + 1;
    adjusted_clock_ = adjusted_clock;

    // TODO: Obtain the era.
    const uint16_t era = 0;

    auto wts = Timestamp::make(era, adjusted_clock, thread_id_);
    wts_.write(wts);

    Timestamp rts = db_->min_wts();
    // Make sure rts < wts; we do not need to worry about collisions by
    // subtracting 1 because (1) every thread does it and (2) timestamp
    // collisions are benign for read-only transactions.
    rts.t2--;
    rts_.write(rts);

    if (for_peek_only_transaction)
      return rts;
    else
      return wts;
  }

  uint64_t allocate_row (Table<StaticConfig>* tbl) {
#if defined (CACHE_QUEUE)
    if (tbl->get_table_as_cache() == false) {
      auto& free_row_ids = free_rows_[tbl];
      if (free_row_ids.empty()) {
        if (!tbl->allocate_rows(this, free_row_ids))
          return static_cast<uint64_t>(-1);
      }
      auto row_id = free_row_ids.back();
      free_row_ids.pop_back();
      if (StaticConfig::kVerbose) printf("new row ID = %" PRIu64 "\n", row_id);
      // gc_ts needs to be initialized with a near-past timestamp.
      auto min_wts = db_->min_wts();
      for (uint16_t cf_id = 0; cf_id < tbl->cf_count(); cf_id++) {
        auto g = tbl->gc_info(cf_id, row_id);
        g->gc_lock = 0;
        g->gc_ts.init(min_wts);
        assert(tbl->head(cf_id, row_id)->older_rv == nullptr);
      }
      return row_id;
    }
    else {
      auto& free_row_ids = free_rows_queue_[tbl];
      if (free_row_ids.empty()) {
        return static_cast<uint64_t>(-1);
      }
      auto row_id = free_row_ids.front ();
      free_row_ids.pop();
      if (StaticConfig::kVerbose) printf("new row ID = %" PRIu64 "\n", row_id);
      // gc_ts needs to be initialized with a near-past timestamp.
      auto min_wts = db_->min_wts();
      for (uint16_t cf_id = 0; cf_id < tbl->cf_count(); cf_id++) {
        auto g = tbl->gc_info(cf_id, row_id);
        g->gc_lock = 0;
        g->gc_ts.init(min_wts);
        assert(tbl->head(cf_id, row_id)->older_rv == nullptr);
      }
      return row_id;
    }
#else
    auto& free_row_ids = free_rows_[tbl];
    if (free_row_ids.empty()) {
      if (!tbl->allocate_rows(this, free_row_ids))
        return static_cast<uint64_t>(-1);
    }
    auto row_id = free_row_ids.back();
    free_row_ids.pop_back();

    if (StaticConfig::kVerbose) printf("new row ID = %" PRIu64 "\n", row_id);

    // gc_ts needs to be initialized with a near-past timestamp.
    auto min_wts = db_->min_wts();
    for (uint16_t cf_id = 0; cf_id < tbl->cf_count(); cf_id++) {
      auto g = tbl->gc_info(cf_id, row_id);
      g->gc_lock = 0;
      g->gc_ts.init(min_wts);

      assert(tbl->head(cf_id, row_id)->older_rv == nullptr);
    }
    return row_id;
#endif
  }  // end of allocate_row

  void deallocate_row(Table<StaticConfig>* tbl, uint64_t row_id) {
#if defined (CACHE_QUEUE)
    if (tbl->get_table_as_cache() == false) {
      free_rows_[tbl].push_back(row_id);
    }
    else {
      free_rows_queue_[tbl].push(row_id);
    }
#else
    free_rows_[tbl].push_back(row_id);
#endif
    // TODO: Defragement and return a group of rows.
  } // end of deallocate_row

  template <bool NewRow>
  RowVersion<StaticConfig>* allocate_version(Table<StaticConfig>* tbl,
                                             uint16_t cf_id, uint64_t row_id,
                                             RowHead<StaticConfig>* head,
                                             uint64_t data_size) {
    #if MEMORY_ADDRESS_WRITE
    address_of_dirty_[address_of_dirty_off_] = (void*)0xf0f0;
    context_pmem_flush (((char*)address_of_dirty_)+(address_of_dirty_off_<<3), 8);
    address_of_dirty_off_ ++;
    #endif

    auto size_cls =
        SharedRowVersionPool<StaticConfig>::data_size_to_class(data_size);

    if (StaticConfig::kInlinedRowVersion && tbl->inlining(cf_id) &&
        size_cls <= tbl->inlined_rv_size_cls(cf_id)) {
      if (!StaticConfig::kInlineWithAltRow && NewRow) {
        assert(head->inlined_rv->status == RowVersionStatus::kInvalid);
        assert(head->inlined_rv->is_inlined());
        // NewRow is guaranteed to have an inlined version available if
        // alternative rows are disabled.
        head->inlined_rv->data_size = static_cast<uint32_t>(data_size);
        return head->inlined_rv;
      } else if (head->inlined_rv->status == RowVersionStatus::kInvalid &&
                 __sync_bool_compare_and_swap(&head->inlined_rv->status,
                                              RowVersionStatus::kInvalid,
                                              RowVersionStatus::kPending)) {
        // Acquire an inlined version by contesting it.
        assert(head->inlined_rv->is_inlined());
        head->inlined_rv->data_size = static_cast<uint32_t>(data_size);
        return head->inlined_rv;
      } else if (StaticConfig::kInlineWithAltRow) {
        auto alt_head = tbl->alt_head(cf_id, row_id);
        if (alt_head->inlined_rv->status == RowVersionStatus::kInvalid &&
            __sync_bool_compare_and_swap(&alt_head->inlined_rv->status,
                                         RowVersionStatus::kInvalid,
                                         RowVersionStatus::kPending)) {
          // Acquire an inlined version at the alternative row by contesting it.
          assert(alt_head->inlined_rv->is_inlined());
          alt_head->inlined_rv->data_size = static_cast<uint32_t>(data_size);
          return alt_head->inlined_rv;
        }
      }
    }

#if defined (test) || defined (CZEN)
    auto pool = db_->row_version_pool(thread_id_);
#elif defined (CWBL)
    auto pool = db_->aep_row_version_pool(thread_id_);
#endif

    auto rv = pool->allocate(size_cls);
    rv->data_size = static_cast<uint32_t>(data_size);
    return rv;
  }

  RowVersion<StaticConfig>* allocate_version_for_new_row(
      Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
      RowHead<StaticConfig>* head, uint64_t data_size) {
    return allocate_version<true>(tbl, cf_id, row_id, head, data_size);
  }

  RowVersion<StaticConfig>* allocate_version_for_existing_row(
      Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
      RowHead<StaticConfig>* head, uint64_t data_size) {
    return allocate_version<false>(tbl, cf_id, row_id, head, data_size);
  }

  void deallocate_version(RowVersion<StaticConfig>* rv) {
    if (StaticConfig::kInlinedRowVersion && rv->is_inlined()) {
      ::mica::util::memory_barrier();
      rv->status = RowVersionStatus::kInvalid;
    } else {


// a lesson, deallocate_version should be consistent with allocate_version
#if defined (test) || defined (CZEN)
      auto pool = db_->row_version_pool(thread_id_);
#elif defined (CWBL)
      auto pool = db_->aep_row_version_pool(thread_id_);
#endif

      pool->deallocate(rv);
    }
  }

  void schedule_gc(/*uint64_t gc_epoch,*/ const Timestamp& wts,
                   Table<StaticConfig>* tbl, uint16_t cf_id, uint8_t deleted,
                   uint64_t row_id, RowHead<StaticConfig>* head,
                   RowVersion<StaticConfig>* write_rv);
  void check_gc();
  void gc(bool forced);

  void quiescence() { db_->quiescence(thread_id_); }
  void idle() { db_->idle(thread_id_); }

  std::vector<uint64_t>& get_free_rows_for_tbl (Table<StaticConfig> *tbl) {
    return free_rows_[tbl];
  }

  size_t get_free_rows_size_for_tbl (Table<StaticConfig> *tbl) {
    return free_rows_[tbl].size();
  }
  
  #if defined (CACHE_QUEUE)
  std::queue<uint64_t>& get_free_rows_queue_for_tbl (Table<StaticConfig> *tbl) {
    return free_rows_queue_[tbl];
  }
  size_t get_free_rows_queue_size_for_tbl (Table<StaticConfig> *tbl) {
    return free_rows_queue_[tbl].size();
  }
  #endif
 
 private:
  friend class Table<StaticConfig>;
  friend class Transaction<StaticConfig>;

  DB<StaticConfig>* db_;
  uint16_t thread_id_;
  uint8_t numa_id_;

  uint16_t next_sync_thread_id_;

  uint64_t last_tsc_;
  uint64_t last_quiescence_;
  uint64_t last_clock_sync_;

  uint64_t clock_boost_;
  uint64_t adjusted_clock_;

  ::mica::util::Rand backoff_rand_;

  std::unordered_map<const Table<StaticConfig>*, std::vector<uint64_t>>
      free_rows_;

  #if defined (CACHE_QUEUE)
  // same as free_rows_ for cache optimazation, change to queue
  std::unordered_map<const Table<StaticConfig>*, std::queue<uint64_t>>
      free_rows_queue_;
  #endif

  struct GCItem {
    // uint64_t gc_epoch;
    Timestamp wts;
    Table<StaticConfig>* tbl;
    uint16_t cf_id;
    uint8_t deleted;  // Move here for better packing.
    uint64_t row_id;
    RowHead<StaticConfig>* head;
    RowVersion<StaticConfig>* write_rv;
  };
  std::queue<GCItem> gc_items_;
  // ::mica::util::SingleThreadedQueue<GCItem, StaticConfig::kMaxGCQueueSize>
  //     gc_items_;

  Stats stats_;
  TimingStack timing_stack_;
  ::mica::util::Latency inter_commit_latency_;
  ::mica::util::Latency commit_latency_;
  ::mica::util::Latency abort_latency_;
  ::mica::util::Latency ro_tx_staleness_;

  // For kPairwiseSleeping.
  uint64_t pair_selector_;

  // Frequently modified by the owner thread, but sometimes read by
  // other threads.
  ConcurrentTimestamp wts_ __attribute__((aligned(64)));
  ConcurrentTimestamp rts_;
  volatile uint64_t clock_;

 private:

  uint64_t clwb_cnt_;
  uint64_t sfence_cnt_;
  uint64_t persist_size_;

 public:
  void context_pmem_persist (const void *p, size_t len) {
    pmem_persist (p, len);

    uint64_t v = reinterpret_cast<uint64_t>(p);
    uint64_t vs = v >>6;
    uint64_t ve = 1+((v+len)>>6);

    clwb_cnt_ += (ve-vs);
    sfence_cnt_ += 1;
    persist_size_ += len;
  }
  void context_pmem_flush (const void *p, size_t len) {
    pmem_flush (p, len);

    uint64_t v = reinterpret_cast<uint64_t>(p);
    uint64_t vs = v >>6;
    uint64_t ve = 1+((v+len)>>6);

    clwb_cnt_ += (ve-vs);
    persist_size_ += len;
  }
  void context_pmem_drain (void) {
    pmem_drain();
    sfence_cnt_ += 1;
  }
  void reset_analysis  (void) {
    persist_size_ = 0;
    clwb_cnt_ = 0;
    sfence_cnt_ = 0;
  }
  uint64_t get_clwb_cnt (void) {
    return clwb_cnt_;
  }
  uint64_t get_sfence_cnt (void) {
    return sfence_cnt_;
  }
  uint64_t get_persist_size (void) {
    return persist_size_;
  }

} __attribute__((aligned(64)));
}
}

#include "context_gc.h"
#endif

