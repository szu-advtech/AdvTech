#pragma once
#ifndef MICA_TRANSACTION_TRANSACTION_H_
#define MICA_TRANSACTION_TRANSACTION_H_

#include "mica/common.h"
#include "mica/transaction/context.h"
#include "mica/transaction/table.h"
#include "mica/transaction/row.h"
#include "mica/transaction/row_access.h"
#include "mica/transaction/timestamp.h"
#include "mica/transaction/stats.h"
#include "mica/util/memcpy.h"

namespace mica {
namespace transaction {
enum class Result {
  kCommitted = 0,
  kAbortedByGetRow,  // Not returned by Transaction::commit() but indicated by
                     // a nullptr return value from get_row_for_write().
  kAbortedByPreValidation,
  kAbortedByDeferredRowVersionInsert,
  kAbortedByMainValidation,
  kAbortedByLogging,
  kInvalid,
};

template <class StaticConfig>
class Transaction {
 public:
  typedef typename StaticConfig::Timing Timing;
  typedef typename StaticConfig::Timestamp Timestamp;

  typedef RowAccessHandle<StaticConfig> RAH;
  typedef RowAccessHandlePeekOnly<StaticConfig> RAHPO;
  typedef PmemRowAccessHandle<StaticConfig> ARAH;
  typedef PmemRowAccessHandlePeekOnly<StaticConfig> ARAHPO;

  static constexpr uint64_t kNewRowID = static_cast<uint64_t>(-1);
  static constexpr uint64_t kDefaultWriteDataSize = static_cast<uint64_t>(-1);

  // transaction_impl/init.h
  Transaction(Context<StaticConfig>* ctx);
  virtual ~Transaction();

  // transaction_impl/commit.h
  bool begin(bool peek_only = false,
             const Timestamp* causally_after_ts = nullptr);

  // transaction_impl/operation.h
  struct NoopDataCopier {
    bool operator()(uint16_t cf_id, RowVersion<StaticConfig>* dest,
                    const RowVersion<StaticConfig>* src) const {
      (void)cf_id;
      (void)dest;
      (void)src;
      return true;
    };
  };
  struct TrivialDataCopier {
    bool operator()(uint16_t cf_id, RowVersion<StaticConfig>* dest,
                    const RowVersion<StaticConfig>* src) const {
      (void)cf_id;
      if (src != nullptr && dest->data_size != 0) {
        assert(dest->data_size >= src->data_size);
        ::mica::util::memcpy(dest->data, src->data, src->data_size);
#if defined (CWBL)
       // index needs no pmem_persist done outside of this function
       // pmem_persist (dest->data, src->data_size); 
       // ctx_->record_persist (src->data_size);
#endif
      }
      return true;
    };
  };

  template <class DataCopier>
  bool new_row(RAH& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
               uint64_t row_id, bool check_dup_access,
               uint64_t data_size, const DataCopier& data_copier);
  void prefetch_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                    uint64_t off, uint64_t len);
  bool peek_row(RAH& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                uint64_t row_id, bool check_dup_access, bool read_hint,
                bool write_hint);
  bool peek_row(RAHPO& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                uint64_t row_id, bool check_dup_access);
  template <class DataCopier>
  bool read_row(RAH& rah, const DataCopier& data_copier);
  template <class DataCopier>
  bool write_row(RAH& rah, uint64_t data_size, const DataCopier& data_copier);
  bool delete_row(RAH& rah);

  // for zen table cache
  bool peek_decide_duplicate (Table<StaticConfig>* tbl,
                              uint16_t cf_id, uint64_t row_id);

  // transaction_impl/commit.h
  struct NoopWriteFunc {
    bool operator()() const { return true; }
  };
  template <class WriteFunc = NoopWriteFunc>
  bool commit(Result* detail = nullptr,
              const WriteFunc& write_func = WriteFunc());
  bool abort (bool skip_backoff = false);

  bool has_began() const { return began_; }
  bool is_peek_only() const { return peek_only_; }

  virtual Context<StaticConfig>* context() { return ctx_; }
  virtual const Context<StaticConfig>* context() const { return ctx_; }

  const Timestamp& ts() const { return ts_; }

  // For logging an verification.
  uint16_t access_size() const { return access_size_; }
  uint16_t iset_size() const { return iset_size_; }
  uint16_t rset_size() const { return rset_size_; }
  uint16_t wset_size() const { return wset_size_; }
  const uint16_t* iset_idx() const { return iset_idx_; }
  const uint16_t* rset_idx() const { return rset_idx_; }
  const uint16_t* wset_idx() const { return wset_idx_; }
  const RowAccessItem<StaticConfig>* accesses() const { return accesses_; }

  // For debugging.
  void print_version_chain(const Table<StaticConfig>* tbl, uint16_t cf_id,
                           uint64_t row_id) const;

 protected:
  // transaction_impl/operation.h
  template <bool ForRead, bool ForWrite, bool ForValidation>
  void locate(RowCommon<StaticConfig>*& newer_rv,
              RowVersion<StaticConfig>*& rv);
  bool insert_version_deferred();
  RowVersionStatus wait_for_pending(RowVersion<StaticConfig>* rv);
  void insert_row_deferred();

  void reserve(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
               bool read_hint, bool write_hint);

  // transaction_impl/commit.h
  Timestamp generate_timestamp();
  void sort_wset();
  bool check_version();
  void update_rts();
  void write();

  void maintenance();
  void backoff();

 protected:
  // transaction_impl/commit.h
  Context<StaticConfig>* ctx_;

  bool began_;
  Timestamp ts_;

  uint16_t access_size_;
  uint16_t iset_size_;
  uint16_t rset_size_;
  uint16_t wset_size_;

  uint8_t consecutive_commits_;

  uint8_t peek_only_;

  uint64_t begin_time_;
  uint64_t* abort_reason_target_count_;
  uint64_t* abort_reason_target_time_;

  uint64_t last_commit_time_;

  uint16_t access_bucket_count_;

  RowAccessItem<StaticConfig> accesses_[StaticConfig::kMaxAccessSize];
  uint16_t iset_idx_[StaticConfig::kMaxAccessSize];
  uint16_t rset_idx_[StaticConfig::kMaxAccessSize];
  uint16_t wset_idx_[StaticConfig::kMaxAccessSize];

  struct AccessBucket {
    static constexpr uint16_t kEmptyBucketID = static_cast<uint16_t>(-1);
    uint16_t count;
    uint16_t next;
    uint16_t idx[StaticConfig::kAccessBucketSize];
  } __attribute__((aligned(64)));
  std::vector<AccessBucket> access_buckets_;

  struct ReserveItem {
    ReserveItem(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                bool read_hint, bool write_hint)
        : tbl(tbl),
          cf_id(cf_id),
          row_id(row_id),
          read_hint(read_hint),
          write_hint(write_hint) {}
    Table<StaticConfig>* tbl;
    uint16_t cf_id;
    uint64_t row_id;
    bool read_hint;
    bool write_hint;
  };
  std::vector<ReserveItem> to_reserve_;

 public:
  void construct_pre_commit_log (void *log_ptr) {
    write_ahead_log_t2 = reinterpret_cast<uint64_t*>(log_ptr); 
  }

  size_t construct_log (char *bf_s, char *bf_e) const;
  bool construct_commit_log (void) const;

 private:
  uint64_t *write_ahead_log_t2;
  size_t construct_write_ahead_entry(char *bf_s, char *bf_e, 
                                     const RowAccessItem<StaticConfig> &item) const;
  size_t construct_write_ahead_log  (char *bf_s, char *bf_e) const;
  size_t construct_write_behind_log (char *bf_s, char *bf_e) const;

 public:
  virtual 
  bool aep_begin (bool peek_only = false) {
    (void)peek_only;
    printf ("Not support aep_begin!\n");
    return false;
  }
  virtual 
  bool aep_commit (Result *result = nullptr) {
    (void)result;
    printf ("Not support aep_commit!\n");
    return false;
  }
  virtual 
  bool aep_abort (bool skip_backoff = false) {
    (void) skip_backoff;
    printf ("Not support aep_abort!\n");
    return false;
  }

  virtual 
  bool aep_read_row (ARAH& rah) {
    (void)rah;
    printf ("Not support aep_read_row!\n");
    return false;
  }
  virtual
  bool aep_write_row (ARAH& rah, uint64_t data_size) {
    (void)rah;
    (void)data_size;
    printf ("Not support aep_write_row!\n");
    return false;
  }
  virtual
  bool aep_peek_row (ARAH& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                     uint64_t row_id, bool check_dup_access, bool read_hint,
                     bool write_hint) {
    (void)rah;
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    (void)read_hint;
    (void)write_hint;
    printf ("Not support aep_peek_row!\n");
    return false;
  }
  virtual
  bool aep_peek_row (ARAHPO& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                     uint64_t row_id, bool check_dup_access) {
    (void)rah;
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    printf ("Not support aep_peek_row (peek_only) !\n");
    return false;
  }
  virtual
  bool aep_new_row  (ARAH& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                     uint64_t row_id, uint64_t prim_key, bool check_dup_access,
                     uint64_t data_size) {
    (void)rah;
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)prim_key;
    (void)check_dup_access;
    (void)data_size;
    printf ("Not support aep_new_row!\n");
    return false;
  }
  virtual
  bool aep_delete_row (ARAH& rah) {
    (void)rah;
    printf ("Not support aep_delete_row!\n");
    return false;
  }

  bool row_is_in_aep (uint64_t row_id) {
    return row_id & StaticConfig::kRowIsInPmemDevice;
  }
  PmemRowVersion<StaticConfig>* transform_2_aep_addr (uint64_t row_id) {
    return reinterpret_cast<PmemRowVersion<StaticConfig>*>
                           (row_id & StaticConfig::kRowIsInPmemMask);
  }
  uint64_t transform_2_u64 (PmemRowVersion<StaticConfig> *aep_rv) {
    return reinterpret_cast<uint64_t>(aep_rv) | StaticConfig::kRowIsInPmemDevice;
  }

};

template <class StaticConfig>
class PmemTransaction : public Transaction<StaticConfig> {
 public:

  PmemTransaction(Context<StaticConfig> *ctx) 
    : Transaction<StaticConfig>(ctx) {}
  ~PmemTransaction () {} 

  typedef RowAccessHandle<StaticConfig> RAH;
  typedef RowAccessHandlePeekOnly<StaticConfig> RAHPO;
  typedef PmemRowAccessHandle<StaticConfig> ARAH;
  typedef PmemRowAccessHandlePeekOnly<StaticConfig> ARAHPO;

  bool aep_begin      (bool peek_only = false);
  bool aep_commit     (Result *result = nullptr);
  bool aep_abort      (bool skip_backoff = false);
 
  // ARAH inhritate RAH
  bool aep_peek_row   (ARAH& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access, bool read_hint,
                       bool write_hint);
  bool aep_peek_row   (ARAHPO& rah, Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access);

  bool aep_read_row   (ARAH& rah);
  bool aep_write_row  (ARAH& rah, uint64_t data_size);

  bool aep_new_row    (ARAH& rah, 
                       Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id, 
                       uint64_t prim_key, bool check_dup_access,
                       uint64_t data_size);
  // must write_row before delete_row
  bool aep_delete_row (ARAH& rah);

 private:
  bool aep_cache_approach_threshold (Table<StaticConfig> *tbl) {
#if defined (CACHE_QUEUE)
    return Transaction<StaticConfig>::ctx_->get_free_rows_queue_size_for_tbl(tbl) 
           < tbl->get_cache_num_threshold_per_thread();
#else
    return Transaction<StaticConfig>::ctx_->get_free_rows_size_for_tbl(tbl) 
           < tbl->get_cache_num_threshold_per_thread();
#endif
  }
  bool aep_modify_primary_index (Table<StaticConfig> *tbl, uint64_t oldv, uint64_t newv);
  bool aep_delete_primary_index (Table<StaticConfig> *tbl, uint64_t oldv);

  bool aep_regulate_cahce_for_table (Table<StaticConfig> *tbl);
  bool aep_commit_item  (void);
  bool aep_abort_item   (void);
};

}
}

#include "transaction_impl/commit.h"
#include "transaction_impl/init.h"
#include "transaction_impl/operation.h"

#endif
