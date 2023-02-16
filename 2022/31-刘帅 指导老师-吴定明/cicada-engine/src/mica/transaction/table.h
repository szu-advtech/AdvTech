#pragma once
#ifndef MICA_TRANSACTION_TABLE_H_
#define MICA_TRANSACTION_TABLE_H_

#include <vector>
#include <queue>

#include "mica/common.h"
#include "mica/transaction/db.h"
#include "mica/transaction/row.h"
#include "mica/transaction/context.h"
#include "mica/transaction/transaction.h"
#include "mica/util/memcpy.h"

namespace mica {
namespace transaction {

template <class StaticConfig>
class DB;

template <class StaticConfig>
class Table {
 public:
  typedef typename StaticConfig::Timestamp Timestamp;
  static constexpr uint64_t kMinimumCacheNum = (1UL<<10);

  Table(DB<StaticConfig>* db,
        uint16_t cf_count,
        const uint64_t* data_size_hints, 
        bool is_index=false,
        bool in_aep=false,
        bool as_cache=false,
        size_t cache_num=0);
  virtual ~Table();

  DB<StaticConfig>* db() { return db_; }
  const DB<StaticConfig>* db() const { return db_; }

  uint16_t cf_count() const { return cf_count_; }

  uint64_t data_size_hint(uint16_t cf_id) const {
    return cf_[cf_id].data_size_hint;
  }

  uint64_t row_count() const { return row_count_; }

  uint8_t inlining(uint16_t cf_id) const { return cf_[cf_id].inlining; }

  uint16_t inlined_rv_size_cls(uint16_t cf_id) const {
    return cf_[cf_id].inlined_rv_size_cls;
  }

  bool is_valid(uint16_t cf_id, uint64_t row_id) const;

  RowHead<StaticConfig>* head(uint16_t cf_id, uint64_t row_id);

  const RowHead<StaticConfig>* head(uint16_t cf_id, uint64_t row_id) const;

  RowHead<StaticConfig>* alt_head(uint16_t cf_id, uint64_t row_id);

  RowGCInfo<StaticConfig>* gc_info(uint16_t cf_id, uint64_t row_id);

  const RowVersion<StaticConfig>* latest_rv(uint16_t cf_id,
                                            uint64_t row_id) const;

  bool allocate_rows(Context<StaticConfig>* ctx,
                     std::vector<uint64_t>& row_ids);

  bool renew_rows(Context<StaticConfig>* ctx, uint16_t cf_id,
                  uint64_t& row_id_begin, uint64_t row_id_end,
                  bool expiring_only);

  template <typename Func>
  bool scan(Transaction<StaticConfig>* tx, uint16_t cf_id, uint64_t off,
            uint64_t len, const Func& f);

  virtual void print_table_status() const;

 protected:
  DB<StaticConfig>* db_;
  uint16_t cf_count_;

  struct ColumnFamilyInfo {
    uint64_t data_size_hint;

    uint64_t rh_offset;

    uint64_t rh_size;
    uint8_t inlining;
    uint16_t inlined_rv_size_cls;
  };

  // We use only the half the first level because of shuffling.
  static constexpr uint64_t kFirstLevelWidth =
      PagePool<StaticConfig>::kPageSize / sizeof(char*) / 2;

  uint64_t total_rh_size_;
  uint64_t second_level_width_;
  uint64_t row_id_shift_;  // log_2(second_level_width)
  uint64_t row_id_mask_;   // (1 << row_id_shift) - 1

  ColumnFamilyInfo cf_[StaticConfig::kMaxColumnFamilyCount];

  char* base_root_;
  char** root_;

  volatile uint32_t lock_ __attribute__((aligned(64)));
  uint64_t row_count_;

  bool is_index_; 
  bool in_aep_;
  bool as_cache_;
  size_t cache_num_;
  size_t cache_num_per_thread_;
  size_t cache_num_threshold_per_thread_;
  size_t victim_mask_;
  size_t victim_base_[StaticConfig::kMaxLCoreCount];
  size_t victim_counter_[StaticConfig::kMaxLCoreCount];

 public:
  bool get_table_is_index () { return is_index_; }
  bool get_table_as_cache () { return as_cache_; }
  bool get_table_in_aep   () { return in_aep_; }
 
  size_t get_next_victim (uint16_t thread_id) {
    while (true) {
      size_t v = get_next_protental_victim (thread_id);
      RowHead<StaticConfig> *h = head(StaticConfig::kPKeyColumnHead, v);
      // uint8_t s = h->status.load();
      uint8_t s = h->status;
      
      // test debug
      // return v;

      // debug for dbx1000
      // printf ("transaction/table.h:get_next_victim thread=%u cache=%lu status=%x\n", thread_id, v, s);

      switch (s) {
        case 0:    return v;
        case 1:    // h->status.store (0);
                   h->status = 0;
                   break;
        // case 0xee: // h->status.store (0);
        //           break;
        case 0xff: break;
      }
    }
  }
  
  size_t get_cache_num_threshold_per_thread () {
    return cache_num_threshold_per_thread_;
  }
  virtual PmemRowVersion<StaticConfig>* aep_allocate_row_version (
            uint16_t cf_id, uint16_t thread_id) {
    (void)cf_id;
    (void)thread_id;
    printf ("Not support aep_allocate_row_version!\n");
    return nullptr;
  }
  virtual void aep_free_row_version (
            uint16_t cf_id, uint16_t thread_id, PmemRowVersion<StaticConfig> *aep_rv) {
    (void)cf_id;
    (void)thread_id;
    (void)aep_rv;
    printf ("Not support aep_free_row_version!\n");
  }

  virtual size_t aep_get_rv_num (void) {
    printf ("Not support aep_get_rv_num!\n");
    return 0;
  }

 private:
  size_t get_next_protental_victim (uint16_t thread_id) {
    size_t victim = victim_base_[thread_id] | (victim_counter_[thread_id] & victim_mask_);
    victim_counter_[thread_id] ++;
    return victim;
  }
 
} __attribute__((aligned(64)));


template<class StaticConfig>
class PmemTable : public Table<StaticConfig> {
 public:
  PmemTable (DB<StaticConfig>* db, uint16_t cf_count,
            const uint64_t* data_size_hints, 
            size_t optional_cache = 0) 
            : Table<StaticConfig>(db, cf_count, data_size_hints, 
                                  false, false, true, optional_cache) {
    for (uint16_t i= 0; i< StaticConfig::kMaxColumnFamilyCount; i ++) {
      for (uint16_t j= 0; j< Table<StaticConfig>::db_->thread_count(); j++) {
        PmemVersionForColumnThread *aep_p = new PmemVersionForColumnThread();
        aep_rvs_[i][j] = aep_p;
      }
    }
  }

  ~PmemTable () {
    PagePool<StaticConfig> *aep_page_pool = Table<StaticConfig>::db_->aep_page_pool();
    for (uint16_t i=0; i< Table<StaticConfig>::cf_count_; i++) {
      for (uint64_t j=0; j< Table<StaticConfig>::db_->thread_count(); j++) {
        for (size_t k=0; k< aep_rvs_[i][j]->pages.size(); k++) {
          auto pg = aep_rvs_[i][j]->pages[k];
          aep_page_pool->free (pg);
        }
      }
    }
    for (uint16_t i= 0; i< StaticConfig::kMaxColumnFamilyCount; i ++) {
      for (uint16_t j= 0; j< Table<StaticConfig>::db_->thread_count(); j++) {
        delete aep_rvs_[i][j];
      }
    }
  }

  PmemRowVersion<StaticConfig>* aep_allocate_row_version (uint16_t cf_id, 
                                                         uint16_t thread_id) {
    auto &free = aep_rvs_[cf_id][thread_id]->free_aep_rvs;
    if (free.empty ()) {
      aep_allocate_row_versions_inside (cf_id, thread_id);
    }
    PmemRowVersion<StaticConfig> *rv = free.front ();
    free.pop ();
    return rv;
  }
  void aep_free_row_version (uint16_t cf_id, uint16_t thread_id, 
                             PmemRowVersion<StaticConfig> *aep_rv) {
    aep_rvs_[cf_id][thread_id]->free_aep_rvs.push (aep_rv);
  }

  size_t aep_get_rv_num (void) {
    size_t page_num = 0;
    size_t free_aep_rv_num = 0;
    size_t sz  = Table<StaticConfig>::data_size_hint(StaticConfig::kPKeyColumnHead)
                 +sizeof(PmemRowVersion<StaticConfig>);

   for (uint16_t i=0; i< Table<StaticConfig>::db_->thread_count(); i++) {
      auto &avfct = aep_rvs_[StaticConfig::kPKeyColumnHead][i];
      auto pagen  = avfct->pages.size();
      auto freen  = avfct->free_aep_rvs.size();
      page_num += pagen;
      free_aep_rv_num += freen;
    }
    
    return page_num*PagePool<StaticConfig>::kPageSize/sz-free_aep_rv_num;
  }

  void print_table_status() const {
    Table<StaticConfig>::print_table_status ();
    printf ("PmemTable Status:\n");

    size_t total_versions = 0;
    size_t total_pages    = 0;
    size_t total_memory   = 0;
    for (uint16_t i=0; i< Table<StaticConfig>::db_->thread_count(); i++) {
      auto &avfct = aep_rvs_[StaticConfig::kPKeyColumnHead][i];
      auto pagen  = avfct->pages.size();

      size_t sz  = Table<StaticConfig>::data_size_hint(StaticConfig::kPKeyColumnHead)
                   +sizeof(PmemRowVersion<StaticConfig>);

      size_t aepvn = (pagen*PagePool<StaticConfig>::kPageSize/sz)
                   > (avfct->free_aep_rvs.size()) 
                   ? (pagen*PagePool<StaticConfig>::kPageSize/sz)
                     - (avfct->free_aep_rvs.size()) 
                   : (avfct->free_aep_rvs.size()) 
                     - (pagen*PagePool<StaticConfig>::kPageSize/sz);

      // printf ("table: thread_id: %8u\taep_versions: %8lu\tfree_aep: %8lu\n", 
      //         i,
      //         pagen*PagePool<StaticConfig>::kPageSize/sz, 
      //         avfct->free_aep_rvs.size());
 
      total_pages += pagen;
      total_versions += aepvn;
      total_memory += aepvn*sz;
      
      printf ("thread_id: %8u\taep_pages: %8lu\taep_versions: %8lu\n", i, pagen ,aepvn);
    }
    printf ("total 2MB pages: %lu\n", total_pages);
    printf ("total aep_versions: %lu\n", total_versions);
    printf ("total memory: %10.3lf MB\n\n", static_cast<double>(total_memory)*0.000001);

/*
    size_t total_pages    = 0;
    for (uint16_t i=0; i< Table<StaticConfig>::db_->thread_count(); i++) {
      auto &avfct = aep_rvs_[StaticConfig::kPKeyColumnHead][i];
      auto pagen  = avfct->pages.size();
      total_pages += pagen;
    }
    printf ("total 2MB pages: %lu\n", total_pages);
    printf ("total approximate memory: %10.3lf MB\n\n", 
             static_cast<double>(total_pages)*PagePool<StaticConfig>::kPageSize*0.000001);
*/

  }

 private:
  
  struct PmemVersionForColumnThread {
    std::vector<char*> pages;
    std::queue<PmemRowVersion<StaticConfig>*> free_aep_rvs;
  };

  PmemVersionForColumnThread*
    aep_rvs_[StaticConfig::kMaxColumnFamilyCount][StaticConfig::kMaxLCoreCount];

  void aep_allocate_row_versions_inside (uint16_t cf_id, uint64_t thread_id) {
    char *pg = Table<StaticConfig>::db_->aep_page_pool()->allocate();
    aep_rvs_[cf_id][thread_id]->pages.push_back (pg);
    size_t sz  = Table<StaticConfig>::data_size_hint(cf_id)
                 +sizeof(PmemRowVersion<StaticConfig>);
    size_t num = PagePool<StaticConfig>::kPageSize / sz;
    char *p=pg, *pend = pg + sz*(num-1);
    while (p <= pend) {
      aep_rvs_[cf_id][thread_id]->free_aep_rvs.push (
        reinterpret_cast<PmemRowVersion<StaticConfig>*>(p));
      p += sz;
    }
  }

  friend class DB<StaticConfig>;  // for zen recovery

} __attribute__((aligned(64)));

}
}

#include "table_impl.h"
#endif
