#pragma once
#ifndef MICA_TRANSACTION_DB_IMPL_H_
#define MICA_TRANSACTION_DB_IMPL_H_

namespace mica {
namespace transaction {

template <class StaticConfig>
DB<StaticConfig>::DB(PagePool<StaticConfig>** page_pools, LoggerInterface* logger,
                     Stopwatch* sw, uint16_t num_threads, uint16_t num_recovery_threads)
    : page_pools_(page_pools),
      logger_(logger),
      sw_(sw),
      num_threads_(num_threads),
      num_recovery_threads_ (num_recovery_threads) {

  assert (page_pools_[0] != nullptr && page_pools_[1] != nullptr);
  page_pool_ = page_pools_[0];
  aep_page_pool_ = page_pools_[1];

  assert(num_threads_ <=
         static_cast<uint16_t>(::mica::util::lcore.lcore_count()));
  assert(num_threads_ <= StaticConfig::kMaxLCoreCount);

  num_numa_ = 0;

  for (uint16_t thread_id = 0; 
       thread_id < std::max(num_threads_, num_recovery_threads_); 
       thread_id++) {
    uint8_t numa_id =
        static_cast<uint8_t>(::mica::util::lcore.numa_id(static_cast<size_t>(thread_id<<1)));

    assert (numa_id == 0);
    if (num_numa_ <= numa_id) num_numa_ = static_cast<uint8_t>(numa_id + 1);
    ctxs_[thread_id] = new Context<StaticConfig>(this, thread_id, numa_id);

    thread_active_[thread_id] = false;
    clock_init_[thread_id] = false;
    thread_states_[thread_id].quiescence = false;
  }

  assert(num_numa_ <= StaticConfig::kMaxNUMACount);
  assert(num_numa_ == 1);

  shared_row_version_pools_[0] =
    new SharedRowVersionPool<StaticConfig>(page_pool_, 0);
  aep_shared_row_version_pools_[0] = 
    new SharedRowVersionPool<StaticConfig>(aep_page_pool_, 0);

  for (uint16_t thread_id = 0; 
       thread_id < std::max(num_threads_, num_recovery_threads_);
       thread_id++) {
    auto pool = new RowVersionPool<StaticConfig>
                    (ctxs_[thread_id], shared_row_version_pools_);
    row_version_pools_[thread_id] = pool;
    auto aep_pool = new RowVersionPool<StaticConfig>
                    (ctxs_[thread_id], aep_shared_row_version_pools_);
    aep_row_version_pools_[thread_id] = aep_pool;
  }

  printf("thread count = %" PRIu16 "\n", num_threads_);
  printf("NUMA count = %" PRIu8 "\n", num_numa_);
  printf("\n");

  last_backoff_print_ = 0;
  last_backoff_update_ = 0;
  backoff_ = 0.;

  active_thread_count_ = 0;
  leader_thread_id_ = static_cast<uint16_t>(-1);

  min_wts_.init(ctxs_[0]->generate_timestamp());
  min_rts_.init(min_wts_.get());
  ref_clock_ = 0;
  // gc_epoch_ = 0;

#if defined (CZEN)
  ts_ckp_.t2_start = reinterpret_cast<uint64_t*>(aep_page_pool_->allocate());
  assert (ts_ckp_.t2_start != nullptr);
  ts_ckp_.t2_next  = ts_ckp_.t2_start;  // uint64_t pointer
  ts_ckp_.t2_end   = ts_ckp_.t2_start+PagePool<StaticConfig>::kPageSize/sizeof(uint64_t);
#endif
}

template <class StaticConfig>
DB<StaticConfig>::~DB() {
  // TODO: Deallocate all rows that are cached in Context before deleting
  // tables.
  for (auto& e : tables_) delete e.second;
  for (auto thread_id = 0; 
       thread_id < std::max(num_threads_, num_recovery_threads_);
       thread_id++) {
    if (row_version_pools_[thread_id] != nullptr) {
      delete row_version_pools_[thread_id];
    }
    if (aep_row_version_pools_[thread_id] != nullptr) {
      delete aep_row_version_pools_[thread_id];
    }
  }
  if (shared_row_version_pools_[0] != nullptr) {
    delete shared_row_version_pools_[0];
  }
  if (aep_shared_row_version_pools_[0] != nullptr) {
    delete aep_shared_row_version_pools_[0];
  }
  for (auto i = 0; i < std::max(num_threads_,num_recovery_threads_); i++) delete ctxs_[i];

#if defined (RECOVERY) && defined (test)
  for (auto i= checkpoint_tbls_.begin(); i!= checkpoint_tbls_.end(); i++) {
    delete i->second;
  }
#endif

#if defined (CZEN)
  if (ts_ckp_.t2_start != nullptr) {
    aep_page_pool_->free(reinterpret_cast<char*>(ts_ckp_.t2_start));
  }
#endif
}

template <class StaticConfig>
bool DB<StaticConfig>::create_table(std::string name, uint16_t cf_count,
                                    const uint64_t* data_size_hints, 
                                    uint64_t expect_row_num,
                                    uint64_t expect_cache_num) {
  printf ("DB<StaticConfig>::create_table name=%s\n", name.c_str());

  if (tables_.find(name) != tables_.end()) return false;

#if defined (test)
  (void)expect_row_num;
  (void)expect_cache_num;
  auto tbl = new Table<StaticConfig>(this, cf_count, data_size_hints,
                                     false, false, false, 0);
#if defined (RECOVERY)
  auto ckp = new PmemCheckpoint<StaticConfig>(aep_page_pool_, cf_count, data_size_hints);
  checkpoint_tbls_[tbl] = ckp;
#endif

#elif defined (CWBL)
  (void)expect_row_num;
  (void)expect_cache_num;
  auto tbl = new Table<StaticConfig>(this, cf_count, data_size_hints,
                                     false, true, false, 0);
#elif defined (CZEN)
  uint64_t regulated_cache = ::mica::util::next_power_of_two (
                             static_cast<uint64_t>(static_cast<double>(expect_row_num)*
                             StaticConfig::kCachePercentageLowBound/num_threads_)
                             )*num_threads_;
  if (expect_cache_num != 0) {
    regulated_cache = ::mica::util::next_power_of_two (
                             static_cast<uint64_t>(
                             expect_cache_num/num_threads_)
                             )*num_threads_;

  }
  auto tbl = new PmemTable<StaticConfig>(this, cf_count, data_size_hints, regulated_cache);
#endif

  tables_[name] = tbl;
  return true;
}

template <class StaticConfig>
bool DB<StaticConfig>::create_hash_index_unique_u64(
    std::string name, Table<StaticConfig>* main_tbl,
    uint64_t expected_row_count) {

  printf ("DB<StaticConfig>::create_hash_index_unique_u64 name=%s\n", name.c_str());

  if (hash_idxs_unique_u64_.find(name) != hash_idxs_unique_u64_.end())
    return false;

  const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
  auto idx = new HashIndexUniqueU64(
      this, main_tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
      expected_row_count);
  hash_idxs_unique_u64_[name] = idx;

  if (index_mp_2_struct_ps[main_tbl].primary_index.type == IndexType::TypeNonExist) {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].primary_index;
    index_struct.type = IndexType::TypeHashIndexUniqueU64;
    index_struct.pointer.hash_unique_u64 = idx;
  }
  else {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].secondary_index
                                [index_mp_2_struct_ps[main_tbl].secondary_index_serial];
    index_struct.type = IndexType::TypeHashIndexUniqueU64;
    index_struct.pointer.hash_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }

  return true;
}

template <class StaticConfig>
bool DB<StaticConfig>::create_hash_index_nonunique_u64(
    std::string name, Table<StaticConfig>* main_tbl,
    uint64_t expected_row_count) {
  if (hash_idxs_nonunique_u64_.find(name) != hash_idxs_nonunique_u64_.end())
    return false;

  const uint64_t kDataSizes[] = {HashIndexNonuniqueU64::kDataSize};
  auto idx = new HashIndexNonuniqueU64(
      this, main_tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
      expected_row_count);
  hash_idxs_nonunique_u64_[name] = idx;

  if (index_mp_2_struct_ps[main_tbl].primary_index.type == IndexType::TypeNonExist) {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].primary_index;
    index_struct.type = IndexType::TypeHashIndexNonUniqueU64;
    index_struct.pointer.hash_non_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }
  else {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].secondary_index
                                [index_mp_2_struct_ps[main_tbl].secondary_index_serial];
    index_struct.type = IndexType::TypeHashIndexNonUniqueU64;
    index_struct.pointer.hash_non_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }

  return true;
}

template <class StaticConfig>
bool DB<StaticConfig>::create_btree_index_unique_u64(
    std::string name, Table<StaticConfig>* main_tbl) {
  if (btree_idxs_unique_u64_.find(name) != btree_idxs_unique_u64_.end())
    return false;

  const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};
  auto idx = new BTreeIndexUniqueU64(
      this, main_tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
  btree_idxs_unique_u64_[name] = idx;

  if (index_mp_2_struct_ps[main_tbl].primary_index.type == IndexType::TypeNonExist) {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].primary_index;
    index_struct.type = IndexType::TypeBtreeIndexUniqueU64;
    index_struct.pointer.btree_unique_u64 = idx;
  }
  else {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].secondary_index
                                [index_mp_2_struct_ps[main_tbl].secondary_index_serial];
    index_struct.type = IndexType::TypeBtreeIndexUniqueU64;
    index_struct.pointer.btree_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }

  return true;
}

template <class StaticConfig>
bool DB<StaticConfig>::create_btree_index_nonunique_u64(
    std::string name, Table<StaticConfig>* main_tbl) {
  if (btree_idxs_nonunique_u64_.find(name) != btree_idxs_nonunique_u64_.end())
    return false;

  const uint64_t kDataSizes[] = {BTreeIndexNonuniqueU64::kDataSize};
  auto idx = new BTreeIndexNonuniqueU64(
      this, main_tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
  btree_idxs_nonunique_u64_[name] = idx;

  if (index_mp_2_struct_ps[main_tbl].primary_index.type == IndexType::TypeNonExist) {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].primary_index;
    index_struct.type = IndexType::TypeBtreeIndexNonUniqueU64;
    index_struct.pointer.btree_non_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }
  else {
    IndexStruct &index_struct = index_mp_2_struct_ps[main_tbl].secondary_index
                                [index_mp_2_struct_ps[main_tbl].secondary_index_serial];
    index_struct.type = IndexType::TypeBtreeIndexNonUniqueU64;
    index_struct.pointer.btree_non_unique_u64 = idx;
    index_mp_2_struct_ps[main_tbl].secondary_index_serial ++;
  }

  return true;
}

template <class StaticConfig>
void DB<StaticConfig>::activate(uint16_t thread_id) {
  // debug
  printf("DB::activate(): thread_id=%hu\n", thread_id);

  if (thread_active_[thread_id]) return;

  if (!clock_init_[thread_id]) {
    // Add one to avoid reusing the same clock value.
    ctxs_[thread_id]->set_clock(ref_clock_ + 1);
    clock_init_[thread_id] = true;
  }
  ctxs_[thread_id]->generate_timestamp();

  // Ensure that no bogus clock/rts is accessed by other threads.
  ::mica::util::memory_barrier();

  thread_active_[thread_id] = true;

  ::mica::util::memory_barrier();

  // auto init_gc_epoch = gc_epoch_;

  ::mica::util::memory_barrier();

  // Keep updating timestamp until it is reflected to min_wts and min_rts.
  while (/*gc_epoch_ - init_gc_epoch < 2 ||*/ min_wts() >
             ctxs_[thread_id]->wts() ||
         min_rts() > ctxs_[thread_id]->rts()) {
    ::mica::util::pause();

    quiescence(thread_id);

    // We also perform clock syncronization to bump up this thread's clock if
    // necessary.
    ctxs_[thread_id]->synchronize_clock();
    ctxs_[thread_id]->generate_timestamp();
  }

  __sync_fetch_and_add(&active_thread_count_, 1);
}

template <class StaticConfig>
void DB<StaticConfig>::deactivate(uint16_t thread_id) {
  // debug
  printf("DB::deactivate(): thread_id=%hu\n", thread_id);

  if (!thread_active_[thread_id]) return;

  // TODO: Clear any garbage collection item in the context.

  // Wait until ref_clock becomes no smaller than this thread's clock.
  // This allows this thread to resume with ref_clock later.
  while (static_cast<int64_t>(ctxs_[thread_id]->clock() - ref_clock_) > 0) {
    ::mica::util::pause();

    quiescence(thread_id);
  }

  thread_active_[thread_id] = false;

  if (leader_thread_id_ == thread_id)
    leader_thread_id_ = static_cast<uint16_t>(-1);

  __sync_sub_and_fetch(&active_thread_count_, 1);
}

template <class StaticConfig>
void DB<StaticConfig>::reset_clock(uint16_t thread_id) {
  assert(!thread_active_[thread_id]);
  clock_init_[thread_id] = false;
}

template <class StaticConfig>
void DB<StaticConfig>::idle(uint16_t thread_id) {
  quiescence(thread_id);

  ctxs_[thread_id]->synchronize_clock();
  ctxs_[thread_id]->generate_timestamp();
}

template <class StaticConfig>
void DB<StaticConfig>::quiescence(uint16_t thread_id) {
  ::mica::util::memory_barrier();

  thread_states_[thread_id].quiescence = true;

  if (leader_thread_id_ == static_cast<uint16_t>(-1)) {
    if (__sync_bool_compare_and_swap(&leader_thread_id_,
                                     static_cast<uint16_t>(-1), thread_id)) {
      last_non_quiescence_thread_id_ = 0;

      auto now = sw_->now();
      last_backoff_update_ = now;
      last_backoff_ = backoff_;
    }
  }

  if (leader_thread_id_ != thread_id) return;

  uint16_t i = last_non_quiescence_thread_id_;
  for (; i < num_threads_; i++)
    if (thread_active_[i] && !thread_states_[i].quiescence) break;
  if (i != num_threads_) {
    last_non_quiescence_thread_id_ = i;
    return;
  }

  last_non_quiescence_thread_id_ = 0;

  bool first = true;
  Timestamp min_wts;
  Timestamp min_rts;

  for (i = 0; i < num_threads_; i++) {
    if (!thread_active_[i]) continue;

    auto wts = ctxs_[i]->wts();
    auto rts = ctxs_[i]->rts();
    if (first) {
      min_wts = wts;
      min_rts = rts;
      first = false;
    } else {
      if (min_wts > wts) min_wts = wts;
      if (min_rts > rts) min_rts = rts;
    }

    thread_states_[i].quiescence = false;
  }

  assert(!first);
  if (!first) {
    // We only increment gc_epoch and update timestamp/clocks when
    // min_rts increases. The equality is required because having a
    // single active thread will make it the same.

    // Ensure wts is no earlier than rts (this can happen if memory ordering is
    // not strict).
    if (min_wts < min_rts) min_wts = min_rts;

    if (min_wts_.get() < min_wts) min_wts_.write(min_wts);

    if (min_rts_.get() <= min_rts) {
      min_rts_.write(min_rts);

      ref_clock_ = ctxs_[thread_id]->clock();
      // gc_epoch_++;
    }
  }
}

template <class StaticConfig>
void DB<StaticConfig>::update_backoff(uint16_t thread_id) {
  if (leader_thread_id_ != thread_id) return;

  uint64_t now = sw_->now();
  uint64_t time_diff = now - last_backoff_update_;

  const uint64_t us = sw_->c_1_usec();

  if (time_diff < StaticConfig::kBackoffUpdateInterval * us) return;

  assert(time_diff != 0);

  uint64_t committed_count = 0;
  for (uint16_t i = 0; i < num_threads_; i++)
    committed_count += ctxs_[i]->stats().committed_count;

  uint64_t committed_diff = committed_count - last_committed_count_;

  double committed_tput =
      static_cast<double>(committed_diff) / static_cast<double>(time_diff);

  double committed_tput_diff = committed_tput - last_committed_tput_;

  double backoff_diff = backoff_ - last_backoff_;

  double new_last_backoff = backoff_;
  double new_backoff = new_last_backoff;

  // If gradient > 0, higher backoff will cause higher tput.
  // If gradient < 0, lower backoff will cause higher tput.
  double gradient;
  if (backoff_diff != 0.)
    gradient = committed_tput_diff / backoff_diff;
  else
    gradient = 0.;

  double incr = StaticConfig::kBackoffHCIncrement * static_cast<double>(us);
  // If we are updating backoff infrequently, we increase a large amount at
  // once.
  incr *= static_cast<double>(time_diff) /
          static_cast<double>(StaticConfig::kBackoffUpdateInterval * us);

  if (gradient < 0)
    new_backoff -= incr;
  else if (gradient > 0)
    new_backoff += incr;
  else {
    if ((now & 1) == 0)
      new_backoff -= incr;
    else
      new_backoff += incr;
  }

  if (new_backoff < StaticConfig::kBackoffMin * static_cast<double>(us))
    new_backoff = StaticConfig::kBackoffMin * static_cast<double>(us);
  if (new_backoff > StaticConfig::kBackoffMax * static_cast<double>(us))
    new_backoff = StaticConfig::kBackoffMax * static_cast<double>(us);

  last_backoff_ = new_last_backoff;
  backoff_ = new_backoff;

  last_backoff_update_ = now;
  last_committed_count_ = committed_count;
  last_committed_tput_ = committed_tput;

  if (StaticConfig::kPrintBackoff &&
      now - last_backoff_print_ >= 100 * 1000 * us) {
    last_backoff_print_ = now;
    printf("backoff=%.3f us\n", backoff_ / static_cast<double>(us));
  }
}

template <class StaticConfig>
void DB<StaticConfig>::reset_backoff() {
  // This requires reset_stats() to be effective.
  backoff_ = 0.;
}

template <class StaticConfig>
void DB<StaticConfig>::make_checkpoint () {
  // here
  printf ("make checkpoint!\n");
  fflush (stdout);
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl = tbl_it->second;
    auto ckp = checkpoint_tbls_[tbl]; // checkpoint for test
    // snap of table
    uint64_t i=0, flag=0;;
    for ( ; i< tbl->row_count(); i++) {
      for (uint16_t cf=0; cf< tbl->cf_count(); cf++) {
        auto hd = tbl->head(cf, i);
        auto rv = const_cast<RowVersion<StaticConfig>*>(tbl->latest_rv(cf, i));

#if defined (DEBUG)
        assert (rv != nullptr);
#endif
        if (rv == nullptr) {
          flag = 1;
          break;
        }

        auto sz = ckp->check_column_family (rv->data, cf, hd->pkey);

        uint64_t v = reinterpret_cast<uint64_t>(rv->data);
        uint64_t vs = v >>6;
        uint64_t ve = 1+((v+sz)>>6);
        db_other_persist_.ckp_clwb_cnt   += (ve-vs);
        db_other_persist_.ckp_sfence_cnt += 1;
        db_other_persist_.ckp_persist_size+= sz;
      }
      if (flag==1) {
        break;
      }
    }
    ckp->row_num_ = i;
  }
}

#if defined (RECOVERY)
template <class StaticConfig>
void DB<StaticConfig>::recovery () {

  printf ("recovery perform\n");
  struct timeval t1,t2;

#if defined (test)
  printf ("Recovery Method: test\n");
  gettimeofday (&t1, nullptr);
  recovery_mmdb ();
#elif defined (CWBL)
  printf ("Recovery Method: CWBL\n");
  gettimeofday (&t1, nullptr);
  recovery_wbl  ();
#elif defined (CZEN)
  printf ("Recovery Method: CZEN\n");
  gettimeofday (&t1, nullptr);
  recovery_zen  ();
#else
  printf ("Recovery Method: ERROR\n");
  gettimeofday (&t1, nullptr);
  printf ("Error: recovery error because of wrong cmake parameter!\n");
#endif

  gettimeofday (&t2, nullptr);
  size_t tf = static_cast<size_t>
    ((t2.tv_sec-t1.tv_sec)*1000000+(t2.tv_usec-t1.tv_usec));
  printf ("recovery time use %7.3lf ms\n", ((double)tf)/1000);

  // exit(0);
}

// Single Table Multiple thread
template <class StaticConfig>
void DB<StaticConfig>::recovery_with_multiple_thread (uint16_t thread_cnt) {
  // Left Undo

  printf ("multi-thread recovery perform\n");
  struct timeval t1,t2;

#if defined (test)
  printf ("Recovery Method: test\n");
  gettimeofday (&t1, nullptr);
  recovery_mmdb_with_multiple_thread(thread_cnt);
#elif defined (CWBL)
  printf ("Recovery Method: CWBL\n");
  gettimeofday (&t1, nullptr);
  recovery_wbl_with_multiple_thread (thread_cnt);
#elif defined (CZEN)
  printf ("Recovery Method: CZEN\n");
  gettimeofday (&t1, nullptr);
  recovery_zen_with_multiple_thread (thread_cnt);
#else
  printf ("Recovery Method: ERROR\n");
  gettimeofday (&t1, nullptr);
  printf ("Error: recovery error because of wrong cmake parameter!\n");
#endif

  printf ("recovery thread number %u threads\n", (unsigned int)thread_cnt);

  gettimeofday (&t2, nullptr);
  size_t tf = static_cast<size_t>
    ((t2.tv_sec-t1.tv_sec)*1000000+(t2.tv_usec-t1.tv_usec));
  printf ("recovery time use %7.3lf ms\n", ((double)tf)/1000);
}

#if defined (test)
template <class StaticConfig>
size_t DB<StaticConfig>::recovery_mmdb_redo_entry (
    char *log_entry_ptr, 
    std::unordered_map<Table<StaticConfig>*, Table<StaticConfig>*> &mp_tbl) {

  // printf ("recovery_mmdb_redo_entry log_entry_ptr=%p\n", log_entry_ptr);

  size_t offset = 0;
  Table<StaticConfig> *tbl = *reinterpret_cast<Table<StaticConfig>**>(log_entry_ptr);
  Table<StaticConfig> *rebuild_tbl = mp_tbl[tbl];

  offset += sizeof(void*);
  uint16_t cf_id = *reinterpret_cast<uint16_t*>(log_entry_ptr+offset);
  offset += sizeof(uint16_t);
  uint64_t row_id = *reinterpret_cast<uint64_t*>(log_entry_ptr+offset);
  offset += sizeof(uint64_t);
  uint64_t pkey = *reinterpret_cast<uint64_t*>(log_entry_ptr+offset);
  (void) pkey;
  offset += sizeof(uint64_t);
  uint8_t intention = *reinterpret_cast<uint8_t*>(log_entry_ptr+offset);
  offset += sizeof(uint8_t);

#if defined (DEBUG)
  printf ("  cf=%u row=%lu pkey=%lu intention=%u\n", cf_id, row_id, pkey, intention);
#endif

  if (intention == 'd') {
    size_t datasize = *reinterpret_cast<size_t*>(log_entry_ptr+offset);
    (void) datasize;
    offset += sizeof(size_t);
 
    RowVersion<StaticConfig> *rv = rebuild_tbl->head(cf_id, row_id)->older_rv;
    rv->status = RowVersionStatus::kDeleted;
    return offset;
  }
  else if (intention == 'c') {
    size_t datasize = *reinterpret_cast<size_t*>(log_entry_ptr+offset);
    offset += sizeof(size_t);
    RowVersion<StaticConfig> *rv = rebuild_tbl->head(cf_id, row_id)->older_rv;
    rv->status = RowVersionStatus::kCommitted; 
    ::mica::util::memcpy (rv->data, log_entry_ptr+offset, datasize);
    return offset + datasize;
  }
  else {
    printf ("recovery_mmdb_redo_entry: intention=%x error!\n", intention);
    exit (-1);
  }
  // no need
  return 0;
}
template <class StaticConfig>
size_t DB<StaticConfig>::recovery_mmdb_redo (
    char *log_ptr, 
    std::unordered_map<Table<StaticConfig>*, Table<StaticConfig>*> &mp_tbl) {
  size_t offset = 0;

  uint64_t t2 = *(reinterpret_cast<uint64_t*>(log_ptr));
  (void) t2;

  offset += sizeof(uint64_t);
  size_t loglen = *(reinterpret_cast<size_t*>(log_ptr+offset));
  offset += sizeof(size_t);
  char *pend = log_ptr+loglen;

#if defined (DEBUG)
  printf ("recovery_mmdb_redo log_ptr=%p t2=%lu loglen=%lu\n", log_ptr, t2, loglen);
#endif
  // exit (0);

  for (char *pentry= log_ptr+offset; pentry<pend; ) {
    size_t len = recovery_mmdb_redo_entry (pentry, mp_tbl);
    pentry += len;
  }
  return loglen;
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_mmdb () {
  // procedure:
  // 1. load checkpoint for table
  // 2. redo log
  // 3. rebuild index

  Context<StaticConfig> *ctx = context();
  // establish table from check point
  std::unordered_map<Table<StaticConfig>*,Table<StaticConfig>*> mp_2_new_tbl;
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl  = tbl_it->second;
    auto cf_count = tbl->cf_count();
    uint64_t row_count = tbl->row_count();
    (void) row_count;
    uint64_t data_size_hints[StaticConfig::kMaxColumnFamilyCount];
    for (uint16_t i=0; i< cf_count; i++) {
      data_size_hints[i] = tbl->data_size_hint(i);      
    }
    auto ntbl = new Table<StaticConfig>(this, cf_count, data_size_hints,
                                        false, false, false, 0);
    mp_2_new_tbl[tbl] = ntbl;
    
    // load checkpoint
    auto ckp = checkpoint_tbls_[tbl];
    uint64_t ckp_count = ckp->row_num_;
    for (uint64_t i=0; i<ckp_count; i++) {
      // allocate row
      uint64_t rid = ctx->allocate_row(ntbl);

#if defined (DEBUG)
      assert(rid == i);
#endif

      for (uint16_t j=0; j<cf_count; j++) {
        // allocate rv
        auto hd = ntbl->head (j, rid);
        auto rv = ctx->allocate_version_for_new_row (ntbl, j, rid, hd, data_size_hints[j]);
        rv->older_rv = nullptr;
        rv->wts.t2   = 0;
        rv->rts.t2 = 0;
        rv->status = RowVersionStatus::kCommitted;
        hd->older_rv = rv;
        // fill data in row

        // printf ("load_checkpoint row=%lu cf=%u pkey=%lu ckp_count=%lu\n" ,i ,j, hd->pkey,ckp_count);

        char *cr = nullptr;
        auto &ckppg = ckp->checkpoint_partition_by_column_family_[j];
        if (ckppg.it_p == nullptr) {
          ckppg.it_p = ckppg.pages[ckppg.it_pg++];
          ckppg.it_pend =
            ckppg.it_p+PagePool<StaticConfig>::kPageSize/ckp->cf_size_[j]*ckp->cf_size_[j];
          cr = ckppg.it_p;
          ckppg.it_p += ckp->cf_size_[j];
        }
        else {
          cr = ckppg.it_p;
          ckppg.it_p += ckp->cf_size_[j];
          if (ckppg.it_p >= ckppg.it_pend) {
            ckppg.it_p = ckppg.pages[ckppg.it_pg++];
            ckppg.it_pend =
              ckppg.it_p+PagePool<StaticConfig>::kPageSize/ckp->cf_size_[j]*ckp->cf_size_[j];
          }
        }
        auto crp = reinterpret_cast<typename PmemCheckpoint<StaticConfig>::CheckRow*>(cr);
        hd->pkey = crp->pkey;
        ::mica::util::memcpy(rv->data, crp->data, data_size_hints[j]);
      }
    }
  }

  printf ("recovery_mmdb: load checkpoint\n");

  // redo log according to transaction timestamp
  // here
  PmemLogger *logger = reinterpret_cast<PmemLogger*>(logger_);

  std::vector<char*>* log_thread_page_array[num_threads_];
  for (uint16_t i=0; i< num_threads_; i++) {
    log_thread_page_array[i] = &(logger->log_partition_by_thread_[i].pages);
  }
  uint32_t current_log_page_count[num_threads_];
  char *current_log_page[num_threads_];
  char *current_log_page_end[num_threads_];
  std::pair<char*, uint16_t> log2page[num_threads_];

  for (uint16_t i= 0; i< num_threads_; i++) {
    current_log_page_count[i] = 1;
    current_log_page[i] = (*log_thread_page_array[i])[0];
    current_log_page_end[i] = current_log_page[i]+PagePool<StaticConfig>::kPageSize;
    log2page[i] = {current_log_page[i], i};
  }
  uint64_t log_thread_count = num_threads_;

  // uint64_t cnt = 0;
  while (log_thread_count != 0) {
    sort (log2page, log2page+log_thread_count, 
          [](std::pair<char*,uint16_t> &a,std::pair<char*,uint16_t> &b) { 
            return *reinterpret_cast<uint64_t*>(a.first) < *reinterpret_cast<uint64_t*>(b.first); 
          });

    char *log = log2page[0].first;
    uint16_t pos = log2page[0].second;

    // printf ("log=%p pos=%u\n", log, pos);
    size_t len = recovery_mmdb_redo (log, mp_2_new_tbl);
    // printf ("cnt=%lu log=%p pos=%u len=%lu t2=%lu\n", cnt++, log, pos, len, *(uint64_t*)log);

    // DEBUG
    if (len == 0) {
      printf ("recovery_mmdb_redo: len==0\n");
      exit(-1);
    }

    if (log+len+sizeof(uint64_t) > current_log_page_end[pos] ||
        *(reinterpret_cast<uint64_t*>(log+len)) == 0UL) {

      if (current_log_page_count[pos] < log_thread_page_array[pos]->size()) {
        current_log_page[pos] = (*log_thread_page_array[pos])[current_log_page_count[pos]++];
        current_log_page_end[pos] = current_log_page[pos] +PagePool<StaticConfig>::kPageSize;
        log2page[0].first = current_log_page[pos];
      }
      else {
        // 没有Log数据页，说明 这个线程没有日志了
        log2page[0] = log2page[log_thread_count-1];
        log_thread_count -= 1;
      }
    }
    else {
      log2page[0].first = log+len;
    }
  }

  printf ("recovery_mmdb: redo serial log\n");

  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;
  // rebuild index 1-> establish index
  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;
    auto isp = i->second;    
    auto isps= isp.primary_index;

    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);

    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();
      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);
      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};
      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  // rebuild index 2->insert index
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl = tbl_it->second;
    auto ntbl = mp_2_new_tbl[tbl];
    auto isps = tbl_2_new_idx[tbl];
   
    for (uint64_t i=0; i< ntbl->row_count(); i++) {
      auto pkey = ntbl->head(StaticConfig::kPKeyColumnHead, i)->pkey;
      if (isps.type == TypeHashIndexUniqueU64) {
        isps.pointer.hash_unique_u64->non_transactional_insert (ctx, pkey, i);
      }
      else if (isps.type == TypeBtreeIndexUniqueU64) {
        isps.pointer.btree_unique_u64->non_transactional_insert(ctx, pkey, i);
      }
      else {
        // this table has no primary index
      }
    }
  }

  printf ("recovery_mmdb: rebuild index\n");
}
#elif defined (CWBL) // end of test
// 1. analyze ts
// 2. restore state, since rebuilding index, traversal with reclaim
template <class StaticConfig>
void DB<StaticConfig>::recovery_wbl  () {
  Context<StaticConfig> *ctx = context();
  PmemLogger *wbl_logger= reinterpret_cast<PmemLogger*>(logger_);

  // 1. find committed ts from each wbl_log
  uint64_t thread_committed_ts[StaticConfig::kMaxLCoreCount];
  for (uint16_t thd_id=0; thd_id< num_threads_; thd_id++) {
    auto &lp = wbl_logger->log_partition_by_thread_[thd_id];
#if defined (DEBUG)
    assert (lp.pages.size()>0);
#endif
    for (size_t pg_num= lp.pages.size(); pg_num !=0; pg_num--) {
      char *pg = lp.pages[pg_num-1];
      char *pgend = pg+PagePool<StaticConfig>::kPageSize;
      for (char *p=pg; p< pgend; p+=8) {
        uint64_t t2= *reinterpret_cast<uint64_t*>(p);
        if (t2 == 0) {
          break; // last page found
        }
        thread_committed_ts[thd_id] = t2;
      }
      if (thread_committed_ts[thd_id] != 0) {
        break;  // found 
      }
    }
  }

  // 2. establish index
  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;

  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;
    auto isp = i->second;    
    auto isps= isp.primary_index;

    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);

    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();
      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);
      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};
      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  // 3. table traversal & index rebuild
  for (auto tbl_it= tables_.begin(); tbl_it != tables_.end(); tbl_it++) {
    auto tbl = tbl_it->second;
    auto isps = tbl_2_new_idx[tbl];

    for (size_t i=0; i< tbl->row_count(); i++) {
      auto hd = tbl->head(StaticConfig::kPKeyColumnHead, i);
      auto rv = hd->older_rv;
      // mark abort version
      while (rv != nullptr) {
        auto t2 = rv->wts.t2;
        uint64_t thd = t2 & ((1UL<<8)-1);  // get thread that alloc this rv
        if (t2 > thread_committed_ts[thd]) {
          // this rv must incomplete commit
          rv->status = RowVersionStatus::kAborted;
          // maybe need no persist because if breakdown when recovery, no influence
          if (StaticConfig::kRecoveryWithStatePersist) {
            pmem_persist (const_cast<RowVersionStatus*>(&(rv->status)), sizeof(rv->status));
          }
        }
        else {
          break;
        }
        rv = rv->older_rv;
      }
      // find committed visiable version
      while (rv != nullptr) {
        if (rv->status == RowVersionStatus::kCommitted ||
            rv->status == RowVersionStatus::kDeleted) {
          break;
        }
        else {
          rv = rv->older_rv;
        }
      }
      // build primark index
      if (rv && rv->status == RowVersionStatus::kCommitted) {
        uint64_t pkey = hd->pkey;
        if (isps.type == TypeHashIndexUniqueU64) {
          isps.pointer.hash_unique_u64->non_transactional_insert (ctx, pkey, i);
        }
        else if (isps.type == TypeBtreeIndexUniqueU64) {
          isps.pointer.btree_unique_u64->non_transactional_insert(ctx, pkey, i);
        }
        else {
          // this table has no primary index
        }
      }
    }
  }
}
#elif defined (CZEN) // end of CWBL
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_index (
                                 Table<StaticConfig>   *tblp,
                                 HashIndexUniqueU64    *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs) {
  for (uint16_t thd_id=0; thd_id< num_threads_; thd_id++) {
    auto tbl = reinterpret_cast<PmemTable<StaticConfig>*>(tblp);
    auto avfct = tbl->aep_rvs_[StaticConfig::kPKeyColumnHead][thd_id];  // pointer
    auto dt_sz = tbl->data_size_hint(StaticConfig::kPKeyColumnHead);
    auto rv_sz = dt_sz+sizeof(PmemRowVersion<StaticConfig>);
    for (auto pg_id=0UL; pg_id< avfct->pages.size(); pg_id++) {
      char *p = avfct->pages[pg_id];
      char *pend = p+PagePool<StaticConfig>::kPageSize/rv_sz*rv_sz;
      for ( ;p< pend; p+= rv_sz) {
        PmemRowVersion<StaticConfig> *rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>(p);
        // compare committed t2 comfirmed
        if (rv->cmark == 1) {
          committed_ts_comfirmed = std::max(committed_ts_comfirmed, rv->wts.t2);
        }
        if (rv->wts.t2<= committed_ts_comfirmed) {
          // committed rv, lookup in index
          uint64_t pkey = rv->pkey;
          uint64_t result[16], pos=0;
          auto func = [&result, &pos] (auto &k, auto &v) {
            (void)k;
            result[pos++] = v;
            return false;
          };
          uint64_t ret_num = idx->non_transactional_lookup_ptr(pkey, func);

          // ret_num decide to insert or update
          if (ret_num == 0) {
            uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
            idx->non_transactional_insert (ctx, pkey, val);
          }
          else {
#if defined (DEBUG)
            assert (ret_num == 1 && pos == 1);
#endif
            // this logical row with pkey, has inserted a version
            for (uint64_t k=0; k< pos; k++) {
              uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
              auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                            (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
              if (rv->wts.t2 < idx_rv->wts.t2) {
                // rv is an older version, garbage it, maybe garbage to a random thread
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
                }
              }
              else {
                // rv is a newer version, idx points to it, garbage idx_rv
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                            ctx->thread_id(), idx_rv);
                }
                uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
                // modify idx
                *reinterpret_cast<uint64_t*>(result[k]) = val;
              }
            }
          }
        }
        else {
          // pending rv
          pending_rvs.push_back (rv);
        }
      }
    } // end of page loop
  } // end of thread loop
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_index (
                                 Table<StaticConfig>   *tblp,
                                 BTreeIndexUniqueU64   *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs) {
  for (uint16_t thd_id=0; thd_id< num_threads_; thd_id++) {
    auto tbl = reinterpret_cast<PmemTable<StaticConfig>*>(tblp);
    auto avfct = tbl->aep_rvs_[StaticConfig::kPKeyColumnHead][thd_id];  // pointer
    auto dt_sz = tbl->data_size_hint(StaticConfig::kPKeyColumnHead);
    auto rv_sz = dt_sz+sizeof(PmemRowVersion<StaticConfig>);
    for (auto pg_id=0UL; pg_id< avfct->pages.size(); pg_id++) {
      char *p = avfct->pages[pg_id];
      char *pend = p+PagePool<StaticConfig>::kPageSize/rv_sz*rv_sz;
      for ( ;p< pend; p+= rv_sz) {
        PmemRowVersion<StaticConfig> *rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>(p);
        // compare committed t2 comfirmed
        if (rv->cmark == 1) {
          committed_ts_comfirmed = std::max(committed_ts_comfirmed, rv->wts.t2);
        }
        if (rv->wts.t2<= committed_ts_comfirmed) {
          // committed rv, lookup in index
          uint64_t pkey = rv->pkey;
          uint64_t result[16], pos=0;
          auto func = [&result, &pos] (auto &k, auto &v) {
            (void)k;
            result[pos++] = v;
            return false;
          };
          uint64_t ret_num = idx->non_transactional_lookup_ptr(pkey, func);

          // ret_num decide to insert or update
          if (ret_num == 0) {
            uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
            idx->non_transactional_insert (ctx, pkey, val);
          }
          else {
#if defined (DEBUG)
            assert (ret_num == 1 && pos == 1);
#endif
            // this logical row with pkey, has inserted a version
            for (uint64_t k=0; k< pos; k++) {
              uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
              auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                            (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
              if (rv->wts.t2 < idx_rv->wts.t2) {
                // rv is an older version, garbage it, maybe garbage to a random thread
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
                }
              }
              else {
                // rv is a newer version, idx points to it, garbage idx_rv
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                            ctx->thread_id(), idx_rv);
                }
                uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
                // modify idx
                *reinterpret_cast<uint64_t*>(result[k]) = val;
              }
            }
          }
        }
        else {
          // pending rv
          pending_rvs.push_back (rv);
        }
      }
    } // end of page loop
  } // end of thread loop
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_pending_index (
                                 Table<StaticConfig> *tbl,
                                 HashIndexUniqueU64    *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs) {
  for (size_t i=0; i<pending_rvs.size(); i++) {
    PmemRowVersion<StaticConfig> *rv = pending_rvs[i];
    if (rv->wts.t2<= committed_ts_comfirmed) {
      // committed rv, lookup in index
      uint64_t pkey = rv->pkey;
      uint64_t result[16], pos=0;
      auto func = [&result, &pos] (auto &k, auto &v) {
        (void)k;
        result[pos++] = v;
        return false;
      };
      uint64_t ret_num = idx->non_transactional_lookup_ptr(pkey, func);

      // ret_num decide to insert or update
      if (ret_num == 0) {
        uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
        idx->non_transactional_insert (ctx, pkey, val);
      }
      else {
#if defined (DEBUG)
        assert (ret_num == 1 && pos == 1);
#endif
        // this logical row with pkey, has inserted a version
        for (uint64_t k=0; k< pos; k++) {
          uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
          auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                        (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
          if (rv->wts.t2 < idx_rv->wts.t2) {
            // rv is an older version, garbage it, maybe garbage to a random thread
            if (StaticConfig::kRecoveryWithGarbage) {
              tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
            }
          }
          else {
            // rv is a newer version, idx points to it, garbage idx_rv
            if (StaticConfig::kRecoveryWithGarbage) {
              tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                        ctx->thread_id(), idx_rv);
            }
            uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
            // modify idx
            *reinterpret_cast<uint64_t*>(result[k]) = val;
          }
        }
      }
    }
    else { // garbage this rv
      if (StaticConfig::kRecoveryWithGarbage) {
        tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
      }
    }
  }
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_pending_index (
                                 Table<StaticConfig> *tbl,
                                 BTreeIndexUniqueU64   *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs) {
  for (size_t i=0; i<pending_rvs.size(); i++) {
    PmemRowVersion<StaticConfig> *rv = pending_rvs[i];
    if (rv->wts.t2<= committed_ts_comfirmed) {
      // committed rv, lookup in index
      uint64_t pkey = rv->pkey;
      uint64_t result[16], pos=0;
      auto func = [&result, &pos] (auto &k, auto &v) {
        (void)k;
        result[pos++] = v;
        return false;
      };
      uint64_t ret_num = idx->non_transactional_lookup_ptr(pkey, func);

      // ret_num decide to insert or update
      if (ret_num == 0) {
        uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
        idx->non_transactional_insert (ctx, pkey, val);
      }
      else {
#if defined (DEBUG)
        assert (ret_num == 1 && pos == 1);
#endif
        // this logical row with pkey, has inserted a version
        for (uint64_t k=0; k< pos; k++) {
          uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
          auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                        (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
          if (rv->wts.t2 < idx_rv->wts.t2) {
            // rv is an older version, garbage it, maybe garbage to a random thread
            if (StaticConfig::kRecoveryWithGarbage) {
              tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
            }
          }
          else {
            // rv is a newer version, idx points to it, garbage idx_rv
            if (StaticConfig::kRecoveryWithGarbage) {
              tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                        ctx->thread_id(), idx_rv);
            }
            uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
            // modify idx
            *reinterpret_cast<uint64_t*>(result[k]) = val;
          }
        }
      }
    }
    else { // garbage this rv
      if (StaticConfig::kRecoveryWithGarbage) {
        tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
      }
    }
  }
}
// steps:
// 1. create new primary index for each table with primary index
// 2. traverse aep versions to rebuild primary index
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen  () {
  Context<StaticConfig> *ctx = context();

  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;
  std::unordered_map<Table<StaticConfig>*, std::vector<PmemRowVersion<StaticConfig>*>> 
    tbl_2_pending_aep;

  size_t expected_pending_num = StaticConfig::kTimestampCheckpointPerContext*num_threads_;

  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;
    tbl_2_pending_aep[tbl].reserve(expected_pending_num);

    auto isp = i->second;    
    auto isps= isp.primary_index;
    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);
    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();
      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);
      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};
      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  // DEBUG
  printf ("recovery_zen: create new index\n");

  // get currentlly comfirmed committed timestamp
  uint64_t committed_ts_comfirmed = ts_ckp_.get_maximum_timestamp ();
  // uint64_t committed_ts_comfirmed = 0;

  // DEBUG
  printf ("recovery_zen: committed_ts_comfirmed\n");

#if defined (DEBUG)
  assert (committed_ts_comfirmed != 0);
#endif

  // rebuild primary index based on committed timestamp
  for (auto i=tbl_2_pending_aep.begin(); i!= tbl_2_pending_aep.end(); i++) {
    auto tbl  = i->first;            // table pointer
    std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs = i->second;
    auto isps = tbl_2_new_idx[tbl];  // hash or btree index pointer

    if (isps.type == TypeHashIndexUniqueU64) {  // decrease overhead
      auto idx = isps.pointer.hash_unique_u64;
      recovery_zen_rebuild_index (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
    } // end of index type hash
    else if (isps.type == TypeBtreeIndexUniqueU64) {  // decrease overhead
      auto idx = isps.pointer.btree_unique_u64;
      recovery_zen_rebuild_index (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
    } // end of type==TypeBtreeIndex
  } // end of for tbl_2_pending_aep

  // DEBUG
  printf ("recovery_zen: rebuild index\n");

  // rebuild with pending rv based on current comfirmed t2 
  for (auto i=tbl_2_pending_aep.begin(); i!= tbl_2_pending_aep.end(); i++) {
    auto tbl  = i->first;            // table pointer
    std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs = i->second;
    auto isps = tbl_2_new_idx[tbl];  // hash or btree index pointer

    if (isps.type == TypeHashIndexUniqueU64) {  // decrease overhead
      auto idx = isps.pointer.hash_unique_u64;
      recovery_zen_rebuild_pending_index (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
    } // end of index type hash
    else if (isps.type == TypeBtreeIndexUniqueU64) {  // decrease overhead
      auto idx = isps.pointer.btree_unique_u64;
      recovery_zen_rebuild_pending_index (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
    } // end of type==TypeBtreeIndex
  } // end of for tbl_2_pending_aep

  // DEBUG
  printf ("recovery_zen: rebuild pending\n");

}
#endif  // end of CZEN

#if defined (test)  // mmdb
template <class StaticConfig>
size_t DB<StaticConfig>::recovery_mmdb_analyze_redoentry (
    char *log_entry_ptr, uint64_t ts,
    std::unordered_map<Table<StaticConfig>*, HashTable> &mp_2_ht) {

  // printf ("recovery_mmdb_redo_entry log_entry_ptr=%p\n", log_entry_ptr);
  size_t offset = 0;
  Table<StaticConfig> *tbl = *reinterpret_cast<Table<StaticConfig>**>(log_entry_ptr);
  HashTable ht = mp_2_ht[tbl];
  offset += sizeof(void*);

  Elem e;
  e.ts = ts;
  e.cf = *reinterpret_cast<uint16_t*>(log_entry_ptr+offset);
  offset += sizeof(uint16_t);
  e.row = *reinterpret_cast<uint64_t*>(log_entry_ptr+offset);
  offset += sizeof(uint64_t);
  uint64_t pkey = *reinterpret_cast<uint64_t*>(log_entry_ptr+offset);
  (void) pkey;
  offset += sizeof(uint64_t);
  e.intention = *reinterpret_cast<uint8_t*>(log_entry_ptr+offset);
  offset += sizeof(uint8_t);

#if defined (DEBUG)
  printf ("  cf=%u row=%lu pkey=%lu intention=%u\n", cf_id, row_id, pkey, intention);
#endif

  e.sz = *reinterpret_cast<size_t*>(log_entry_ptr+offset);
  offset += sizeof(size_t);
  e.data = log_entry_ptr+offset;

  ht.add (e.row, e);
 
  if (e.intention == 'd') {
    return offset;
  }
  else if (e.intention == 'c') {
    return offset + e.sz;
  }
  else {
    printf ("recovery_mmdb_analyze_redoentry: intention=%x error!\n", e.intention);
    exit (-1);
  }
  // no need
  return 0;
}
template <class StaticConfig>
size_t DB<StaticConfig>::recovery_mmdb_analyze_redolog (
    char *log_ptr, 
    std::unordered_map<Table<StaticConfig>*, HashTable> &mp_2_ht) {
  size_t offset = 0;

  uint64_t t2 = *(reinterpret_cast<uint64_t*>(log_ptr));
  offset += sizeof(uint64_t);
  size_t loglen = *(reinterpret_cast<size_t*>(log_ptr+offset));
  offset += sizeof(size_t);
  char *pend = log_ptr+loglen;

#if defined (DEBUG)
  printf ("recovery_mmdb_analyze_redolog log_ptr=%p t2=%lu loglen=%lu\n", log_ptr, t2, loglen);
#endif

  for (char *pentry= log_ptr+offset; pentry<pend; ) {
    size_t len = 0;
    len = recovery_mmdb_analyze_redoentry (pentry, t2, mp_2_ht);
    pentry += len;
  }
  return loglen;
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_mmdb_with_multiple_thread(uint16_t thread_cnt) {
  // procedure:
  // 1. load checkpoint for table
  // 2. redo log
  // 3. rebuild index

  struct timeval net_t1, net_t2, net_t3, net_t4, net_t5, net_t6, net_t7;
  gettimeofday (&net_t1, nullptr);

  // establish table from check point
  std::unordered_map<Table<StaticConfig>*,Table<StaticConfig>*> mp_2_new_tbl;
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl  = tbl_it->second;
    auto cf_count = tbl->cf_count();
    uint64_t row_count = tbl->row_count();
    (void) row_count;
    uint64_t data_size_hints[StaticConfig::kMaxColumnFamilyCount];
    for (uint16_t i=0; i< cf_count; i++) {
      data_size_hints[i] = tbl->data_size_hint(i);      
    }
    auto ntbl = new Table<StaticConfig>(this, cf_count, data_size_hints,
                                        false, false, false, 0);
    mp_2_new_tbl[tbl] = ntbl;
  }

  // free memory taken by old table
  for (auto i=mp_2_new_tbl.begin(); i!= mp_2_new_tbl.end(); i++) {
    Table<StaticConfig> *tbl = i->first;
    tbl->~Table<StaticConfig>();
  }

  auto recovery_checkpoint = [&] (uint16_t thd_id) {
    Context<StaticConfig> *ctx = context(thd_id);
    ::mica::util::lcore.pin_thread(thd_id);
 
    for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
      auto tbl  = tbl_it->second;
      auto cf_count = tbl->cf_count();
      uint64_t row_count = tbl->row_count();
      (void) row_count;
      uint64_t data_size_hints[StaticConfig::kMaxColumnFamilyCount];
      for (uint16_t i=0; i< cf_count; i++) {
        data_size_hints[i] = tbl->data_size_hint(i);      
      }
      auto ntbl = mp_2_new_tbl[tbl];
  
      // load checkpoint
      auto ckp = checkpoint_tbls_[tbl];
      uint64_t ckp_count = ckp->row_num_;

      uint64_t i_len   = ckp_count / thread_cnt;
      uint64_t i_start = thd_id * i_len;
      uint64_t i_end   = i_start + i_len;
      if (thd_id+1 == thread_cnt) {
        i_end = ckp_count;
      }

      char  *ckp_p[cf_count], *ckp_pend[cf_count];
      size_t ckp_page_serial[cf_count];
      for (uint16_t j= 0; j< cf_count; j++) {
        uint64_t cfdata_per_page = PagePool<StaticConfig>::kPageSize/ckp->cf_size_[j];
        ckp_page_serial[j] = i_start / cfdata_per_page;
        ckp_p[j]    = ckp->checkpoint_partition_by_column_family_[j].pages[ckp_page_serial[j]]
                      + (i_start % cfdata_per_page) * ckp->cf_size_[j];
        ckp_pend[j] = ckp->checkpoint_partition_by_column_family_[j].pages[ckp_page_serial[j]]
                      + PagePool<StaticConfig>::kPageSize/ckp->cf_size_[j]*ckp->cf_size_[j];
      }

      for (uint64_t i=i_start; i<i_end; i++) {
        // allocate row
        uint64_t rid = ctx->allocate_row(ntbl);
  
        for (uint16_t j=0; j<cf_count; j++) {
          // allocate rv
          auto hd = ntbl->head (j, rid);
          auto rv = ctx->allocate_version_for_new_row (ntbl, j, rid, hd, data_size_hints[j]);
          rv->older_rv = nullptr;
          rv->wts.t2   = 0;
          rv->rts.t2 = 0;
          rv->status = RowVersionStatus::kCommitted;
          hd->older_rv = rv;

          // fill data in row  
          // printf ("load_checkpoint row=%lu cf=%u pkey=%lu ckp_count=%lu\n" ,i ,j, hd->pkey,ckp_count);
  
          char *cr = ckp_p[j];
          auto crp = reinterpret_cast<typename PmemCheckpoint<StaticConfig>::CheckRow*>(cr);
          hd->pkey = crp->pkey;
          ::mica::util::memcpy(rv->data, crp->data, data_size_hints[j]);

          if (cr+ckp->cf_size_[j] < ckp_pend[j]) {
            ckp_p[j] += ckp->cf_size_[j];
          }
          else {
            ckp_page_serial[j] += 1;
            ckp_p[j] = ckp->checkpoint_partition_by_column_family_[j].pages[ckp_page_serial[j]];
            ckp_pend[j] = 
              ckp_p[j]+PagePool<StaticConfig>::kPageSize/ckp->cf_size_[j]*ckp->cf_size_[j];
          }
        }
      }
    }
  };

  gettimeofday (&net_t2, nullptr);
  uint16_t thd = 1;
  std::vector<std::thread> threads;
  for ( ; thd < thread_cnt; thd ++) {
    threads.emplace_back (recovery_checkpoint, thd);
  }
  recovery_checkpoint (0);
  while (threads.size() > 0) {
    threads.back().join ();
    threads.pop_back ();
  }
  gettimeofday (&net_t3, nullptr);

  printf ("recovery_mmdb_with_multiple_recovery: load checkpoint\n");
  
  // free memory taken by checkpoint
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl = tbl_it->second;
    PmemCheckpoint<StaticConfig> *ckp = checkpoint_tbls_[tbl];
    ckp->~PmemCheckpoint();
  }

  // analyze redo log according to row id
  std::unordered_map<Table<StaticConfig>*, HashTable> mp_2_ht;
  for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
    auto tbl  = tbl_it->second;
    // auto cf_count = tbl->cf_count();
    uint64_t row_count = tbl->row_count();
    HashTable ht;
    ht.init (row_count, (uint64_t)(0.2*(double)row_count));
    mp_2_ht[tbl] = ht;
  }

  PmemLogger *logger = reinterpret_cast<PmemLogger*>(logger_);
  volatile uint16_t running_threads = 0;
  auto recovery_redolog = [&] (uint16_t thd_id) {
    // part 1 insert defuse redolog and insert into hashtable
    for (uint16_t j=0; j< StaticConfig::kMaxLCoreCount; j++) {
      auto &partition = logger->log_partition_by_thread_[j];
      auto pg_len = partition.pages.size();
      auto thd_len = pg_len / thread_cnt;
      auto i_start = thd_id * thd_len;
      auto i_end   = i_start+ thd_len;
      if (thd_id+1 == thread_cnt) {
        i_end = pg_len;
      }
      for (auto i=i_start; i< i_end; i++) {
        char *p = partition.pages[i];
        char *pend = p+PagePool<StaticConfig>::kPageSize;

        while (p < pend) {

          // printf ("thd=%u page=%p logaddr=%p\n", thd_id, partition.pages[i], p);

          size_t offset = 0;
          offset = recovery_mmdb_analyze_redolog (p, mp_2_ht);

          // printf ("thd=%u page=%p logaddr=%p %lu\n", thd_id, partition.pages[i], p, offset); 
          // fflush (stdout);

          if (offset == 0) {
            break;
          }
          // assert (offset !=0);

          p += offset;
          if (p+sizeof(uint64_t)>pend || *reinterpret_cast<uint64_t*>(p) == 0) {
            break;
          }
        }
      }
    }

    // printf ("finish analysis and start redo -1 thd_id=%u\n", (unsigned int)thd_id);

    __sync_add_and_fetch(&running_threads, 1);
    while (running_threads < thread_cnt) ::mica::util::pause();

    // printf ("finish analysis and start redo -2 thd_id=%u\n", (unsigned int)thd_id);
    // fflush (stderr);

    // part 2 redo log according to latest log
    for (auto i=mp_2_new_tbl.begin(); i!= mp_2_new_tbl.end(); i++) {
      auto tbl = i->first;
      auto ntbl= i->second;
      auto ht  = mp_2_ht[tbl];
      // auto cfs = ntbl->cf_count();
      
      uint64_t j_len = ht.ht_cell_num / thread_cnt;
      uint64_t j_start = j_len * thd_id;
      uint64_t j_end   = j_start + j_len;
      if (thd_id+1  == thread_cnt) {
        j_end = ht.ht_cell_num;
      }
      for (auto j= j_start; j< j_end; j++) {
        HashCell &cell = ht.ht_cell[j];

        // printf ("j_start=%lu j_end=%lu thd_id=%u j=%lu hc_num=%d\n", j_start, j_end, (unsigned int)thd_id, j, cell.hc_num);
        // fflush (stdout);

        if (cell.hc_num == 1) {
          Elem &elem = cell.hc_union.hc_elem.elem;
          if (elem.intention == 'c') {

            // printf ("j_start=%lu j_end=%lu thd_id=%u j=%lu hc_num=%d ts=%lu row=%lu cf=%x intention=%x sz=%lu, data=%p\n", j_start, j_end, (unsigned int)thd_id, j, cell.hc_num, elem.ts, elem.row, elem.cf, elem.intention, elem.sz, elem.data);
            // fflush (stdout);

            // while (ntbl->head(elem.cf, elem.row) == nullptr) printf ("head=nullptr\n");
            // while (ntbl->head(elem.cf, elem.row)->older_rv == nullptr) printf ("rv=nullptr\n");

            RowVersion<StaticConfig> *rv = ntbl->head(elem.cf, elem.row)->older_rv;
            if (rv == nullptr) {
              continue;
            }
            rv->status = RowVersionStatus::kCommitted;
            ::mica::util::memcpy (rv->data, elem.data, elem.sz);
          }
          else if (elem.intention == 'd') {
            RowVersion<StaticConfig> *rv = ntbl->head(elem.cf, elem.row)->older_rv;
            if (rv == nullptr) {
              continue;
            }
            rv->status = RowVersionStatus::kDeleted;
          }
          else {
            printf ("intention error! db_impl.h\n");
            assert (false);
            exit(-1);
          }
        }
        else if (cell.hc_num > 1) {
          HashElem *elems = cell.hc_union.hc_elems.elems;
          // HERE, Asume CF=1
          int select = 0;
          uint64_t ts= 0;
          for (int k=0; k< cell.hc_num; k++) {
            if (elems[k].elem.ts > ts) {
              ts = elems[k].elem.ts;
              select = k;
            }
          }
          Elem &elem = elems[select].elem;

          // printf ("j_start=%lu j_end=%lu thd_id=%u j=%lu hc_num=%d ts=%lu row=%lu cf=%x intention=%x sz=%lu, data=%p\n", j_start, j_end, (unsigned int)thd_id, j, cell.hc_num, elem.ts, elem.row, elem.cf, elem.intention, elem.sz, elem.data);
          // fflush (stdout);

          if (elem.intention == 'c') {

            // while (ntbl->head(elem.cf, elem.row) == nullptr) printf ("head=nullptr\n");
            // while (ntbl->head(elem.cf, elem.row)->older_rv == nullptr) printf ("rv=nullptr\n");

            RowVersion<StaticConfig> *rv = ntbl->head(elem.cf, elem.row)->older_rv;
            if (rv == nullptr) {
              continue;
            }
            rv->status = RowVersionStatus::kCommitted;
            ::mica::util::memcpy (rv->data, elem.data, elem.sz);
          }
          else if (elem.intention == 'd') {
            RowVersion<StaticConfig> *rv = ntbl->head(elem.cf, elem.row)->older_rv;
            if (rv == nullptr) {
              continue;
            }
            rv->status = RowVersionStatus::kDeleted;
          }
          else {
            printf ("intention error! db_impl.h\n");
            assert (false);
            exit(-1);
          }

        }
      }
    }
  };

  {
    gettimeofday (&net_t4, nullptr);
    auto thd = 1;
    std::vector<std::thread> threads;
    for ( ; thd < thread_cnt; thd ++) {
      threads.emplace_back (recovery_redolog, thd);
    }
    recovery_redolog (0);
    while (threads.size() > 0) {
      threads.back().join ();
      threads.pop_back ();
    }
    gettimeofday (&net_t5, nullptr);
    printf ("recovery_mmdb: redo serial log for table\n");
  }
  // time should not include
  for (auto i= mp_2_ht.begin(); i!=mp_2_ht.end(); i++) {
    auto ht = i->second;
    ht.destroy ();
  }
 
  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;
  // rebuild index 1-> establish index
  Context<StaticConfig> *ctx = context();
  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;
    auto isp = i->second;    
    auto isps= isp.primary_index;

    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);

    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();

      // free memory taken by older index
      isps.pointer.hash_unique_u64->idx_tbl_->~Table<StaticConfig>();

      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);
      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};

      // free memory taken by older index 
      isps.pointer.btree_unique_u64->idx_tbl_->~Table<StaticConfig>();
 
      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));
      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  auto recovery_index = [&](uint16_t thd_id) {
    ::mica::util::lcore.pin_thread(thd_id);
    for (auto tbl_it= tables_.begin(); tbl_it!= tables_.end(); tbl_it++) {
      auto tbl = tbl_it->second;
      auto ntbl = mp_2_new_tbl[tbl];
      auto isps = tbl_2_new_idx[tbl];

      uint64_t i_len   = ntbl->row_count()/thread_cnt;
      uint64_t i_start = thd_id * i_len;
      uint64_t i_end   = i_start+ i_len;
      if (thd_id+1 == thread_cnt) {
        i_end = ntbl->row_count();
      }
     
      for (uint64_t i=i_start; i< i_end; i++) {
        auto pkey = ntbl->head(StaticConfig::kPKeyColumnHead, i)->pkey;
        if (isps.type == TypeHashIndexUniqueU64) {
          isps.pointer.hash_unique_u64->non_transactional_insert (ctx, pkey, i);
        }
        else if (isps.type == TypeBtreeIndexUniqueU64) {
          isps.pointer.btree_unique_u64->non_transactional_insert(ctx, pkey, i);
        }
        else {
          // this table has no primary index
        }
      }
    }
  };

  {
    gettimeofday (&net_t6, nullptr);
    auto thd = 1;
    std::vector<std::thread> threads;
    for ( ; thd < thread_cnt; thd ++) {
      threads.emplace_back (recovery_index, thd);
    }
    recovery_index (0);
    while (threads.size() > 0) {
      threads.back().join ();
      threads.pop_back ();
    }
    gettimeofday (&net_t7, nullptr);
    printf ("recovery_mmdb: rebuild index\n");
  }

  uint64_t net_use_1 = static_cast<uint64_t>((net_t3.tv_sec-net_t2.tv_sec)*1000000+(net_t3.tv_usec-net_t2.tv_usec));
  uint64_t net_use_2 = static_cast<uint64_t>((net_t5.tv_sec-net_t4.tv_sec)*1000000+(net_t5.tv_usec-net_t4.tv_usec));
  uint64_t net_use_3 = static_cast<uint64_t>((net_t7.tv_sec-net_t6.tv_sec)*1000000+(net_t7.tv_usec-net_t6.tv_usec));
  uint64_t net_time_use = net_use_1 + net_use_2 + net_use_3;

  printf ("net time use: %7.3lf ms\n", static_cast<double>(net_time_use)/1000.0);
  printf ("-- mmdb load checkpoint:      %7.3lf ms\n", 
              static_cast<double>(net_use_1)/1000.0);
  printf ("-- mmdb redo data:            %7.3lf ms\n", 
              static_cast<double>(net_use_2)/1000.0);
  printf ("-- mmdb rebuild index:        %7.3lf ms\n",
              static_cast<double>(net_use_3)/1000.0);
}
#elif defined (CWBL) // end of mmdb multiple recovery
template <class StaticConfig>
void DB<StaticConfig>::recovery_wbl_with_multiple_thread (uint16_t thread_cnt) {

  struct timeval net_t1, net_t2, net_t3, net_t4;
  gettimeofday (&net_t1, nullptr);

  Context<StaticConfig> *ctx = context();
  PmemLogger *wbl_logger= reinterpret_cast<PmemLogger*>(logger_);

  // 1. find committed ts from each wbl_log
  uint64_t thread_committed_ts[StaticConfig::kMaxLCoreCount];
  for (uint16_t thd_id=0; thd_id< num_threads_; thd_id++) {
    auto &lp = wbl_logger->log_partition_by_thread_[thd_id];
#if defined (DEBUG)
    assert (lp.pages.size()>0);
#endif
    for (size_t pg_num= lp.pages.size(); pg_num !=0; pg_num--) {
      char *pg = lp.pages[pg_num-1];
      char *pgend = pg+PagePool<StaticConfig>::kPageSize;
      for (char *p=pg; p< pgend; p+=8) {
        uint64_t t2= *reinterpret_cast<uint64_t*>(p);
        if (t2 == 0) {
          break; // last page found
        }
        thread_committed_ts[thd_id] = t2;
      }
      if (thread_committed_ts[thd_id] != 0) {
        break;  // found 
      }
    }
  }
  gettimeofday (&net_t2, nullptr);

  // 2. establish index
  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;

  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;
    auto isp = i->second;    
    auto isps= isp.primary_index;

    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);

    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();
      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);

      // free memory taken by older index
      isps.pointer.hash_unique_u64->idx_tbl_->~Table<StaticConfig>();

      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};
      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));

      // free memory taken by older index
      isps.pointer.btree_unique_u64->idx_tbl_->~Table<StaticConfig>();

      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  auto recovery_func = [&] (uint16_t thd_rank) {

    Context<StaticConfig> *ctx = context(thd_rank);
    ::mica::util::lcore.pin_thread(thd_rank);
 
    // 3. table traversal & index rebuild
    for (auto tbl_it= tables_.begin(); tbl_it != tables_.end(); tbl_it++) {
      auto tbl = tbl_it->second;
      auto isps = tbl_2_new_idx[tbl];

      uint64_t r_cnt   = tbl->row_count();
      uint64_t t_cnt   = r_cnt / thread_cnt;
      uint64_t i_start = thd_rank * t_cnt;
      uint64_t i_end   = i_start + t_cnt;
      if (thd_rank+1 == thread_cnt) {
        i_end = r_cnt;
      }
  
      for (size_t i= i_start; i< i_end; i++) {
        auto hd = tbl->head(StaticConfig::kPKeyColumnHead, i);
        auto rv = hd->older_rv;
        // mark abort version
        while (rv != nullptr) {
          auto t2 = rv->wts.t2;
          uint64_t thd = t2 & ((1UL<<8)-1);  // get thread that alloc this rv
          if (t2 > thread_committed_ts[thd]) {
            // this rv must incomplete commit
            rv->status = RowVersionStatus::kAborted;
            // maybe need no persist because if breakdown when recovery, no influence
            if (StaticConfig::kRecoveryWithStatePersist) {
              pmem_persist (const_cast<RowVersionStatus*>(&(rv->status)), sizeof(rv->status));
            }
          }
          else {
            break;
          }
          rv = rv->older_rv;
        }
        // find committed visiable version
        while (rv != nullptr) {
          if (rv->status == RowVersionStatus::kCommitted ||
              rv->status == RowVersionStatus::kDeleted) {
            break;
          }
          else {
            rv = rv->older_rv;
          }
        }
        // build primark index
        if (rv && rv->status == RowVersionStatus::kCommitted) {
          uint64_t pkey = hd->pkey;
          if (isps.type == TypeHashIndexUniqueU64) {
            isps.pointer.hash_unique_u64->non_transactional_insert (ctx, pkey, i);
          }
          else if (isps.type == TypeBtreeIndexUniqueU64) {
            isps.pointer.btree_unique_u64->non_transactional_insert(ctx, pkey, i);
          }
          else {
            // this table has no primary index
          }
        }
      }
    }
  }; // end of recovery_func 

  gettimeofday (&net_t3, nullptr);
  uint16_t thd = 1;
  std::vector<std::thread> threads;
  for ( ; thd < thread_cnt; thd ++) {
    threads.emplace_back (recovery_func, thd);
  }
  recovery_func (0);
  while (threads.size() > 0) {
    threads.back().join ();
    threads.pop_back ();
  }
  gettimeofday (&net_t4, nullptr);

  uint64_t net_use_1 = (uint64_t)((net_t2.tv_sec-net_t1.tv_sec)*1000000+(net_t2.tv_usec-net_t1.tv_usec));
  uint64_t net_use_2 = (uint64_t)((net_t4.tv_sec-net_t3.tv_sec)*1000000+(net_t4.tv_usec-net_t3.tv_usec));
  uint64_t net_time_use = net_use_1 + net_use_2;

  printf ("net time use: %7.3lf ms\n", static_cast<double>(net_time_use)/1000.0);
  printf ("-- wbl identify timestamp:      %7.3lf ms\n", 
              static_cast<double>(net_use_1)/1000.0);
  printf ("-- wbl recovery data and index: %7.3lf ms\n", 
              static_cast<double>(net_use_2)/1000.0);
  uint64_t trivial_use = (uint64_t)((net_t3.tv_sec-net_t2.tv_sec)*1000000+(net_t3.tv_usec-net_t2.tv_usec));
  printf ("trivial time use: %7.3lf ms\n", static_cast<double>(trivial_use)/1000.0);
}
#elif defined (CZEN) // end of wbl multiple recovery
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_index_with_multiple_thread (
                                 Table<StaticConfig>   *tblp,
                                 HashIndexUniqueU64    *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs,
                                 uint16_t thread_id,
                                 uint16_t thread_cnt) {
  uint64_t local_committed_ts_comfirmed = committed_ts_comfirmed;
  for (uint16_t thd_id=0; thd_id < num_threads_; thd_id++) {
    auto tbl = reinterpret_cast<PmemTable<StaticConfig>*>(tblp);
    auto avfct = tbl->aep_rvs_[StaticConfig::kPKeyColumnHead][thd_id];  // pointer
    auto dt_sz = tbl->data_size_hint(StaticConfig::kPKeyColumnHead);
    auto rv_sz = dt_sz+sizeof(PmemRowVersion<StaticConfig>);

    size_t pg_thd = avfct->pages.size()/thread_cnt;
    size_t pg_start  = pg_thd*thread_id;
    size_t pg_end = pg_start+pg_thd;
    if (thread_id+1 == thread_cnt) {
      pg_end = avfct->pages.size();
    }

    for (auto pg_id=pg_start; pg_id< pg_end; pg_id++) {
      char *p = avfct->pages[pg_id];
      char *pend = p+PagePool<StaticConfig>::kPageSize/rv_sz*rv_sz;
      for ( ;p< pend; p+= rv_sz) {

        // printf ("dealing with a aep_rv!\n");

        PmemRowVersion<StaticConfig> *rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>(p);
        // compare committed t2 comfirmed
        if (rv->cmark == 1) {
          local_committed_ts_comfirmed = std::max(local_committed_ts_comfirmed, rv->wts.t2);
        }
        if (rv->wts.t2 <= local_committed_ts_comfirmed) {
          // committed rv, lookup in index
          uint64_t pkey = rv->pkey;
          uint64_t result[16], pos=0;
          auto func = [&result, &pos] (auto &k, auto &v) {
            (void)k;
            result[pos++] = v;
            return false;
          };

          volatile uint8_t  *cas_status = nullptr;
          uint64_t ret_num = idx->non_transactional_lookup_ptr_for_multiple_recovery
                             (pkey, func, &cas_status);

          // printf ("non_transactional_lookup_ptr_for_multiple_recovery pkey=%lu ret_num=%lu cas_status=%p!\n", pkey, ret_num, cas_status);

          // ret_num decide to insert or update
          if (ret_num == 0) {

            // printf ("insert start\n");

            uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
            idx->non_transactional_insert (ctx, pkey, val);

            // printf ("insert finish\n");
          }
          else {
#if defined (DEBUG)
            printf ("assert\n");
            assert (ret_num == 1 && pos == 1);
#endif
            // this logical row with pkey, has inserted a version
            for (uint64_t k=0; k< pos; k++) {
              uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
              auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                            (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
              if (rv->wts.t2 < idx_rv->wts.t2) {
                // rv is an older version, garbage it, maybe garbage to a random thread
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead,ctx->thread_id(),rv);
                }
              }
              else {
                // rv is a newer version, idx points to it, garbage idx_rv
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                            ctx->thread_id(), idx_rv);
                }
                uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
                // modify idx
                *reinterpret_cast<uint64_t*>(result[k]) = val;
              }
            }

            // printf ("for k=0 loop\n");

          }

          // cas unlock
          // printf ("cas_status=%p\n", cas_status);
          // printf ("*cas_status=%d\n",  (int)(*cas_status));
          // fflush(stdout); 

          *cas_status = 0;
          ::mica::util::memory_barrier();

          // printf ("cas_status unlock end\n");
        }
        else {
          // pending rv
          pending_rvs.push_back (rv);
        }
      }
    } // end of page loop
  } // end of thread loop

  // printf ("thread end of loop\n");

  while (local_committed_ts_comfirmed > committed_ts_comfirmed) {
    if (__sync_bool_compare_and_swap (&committed_ts_comfirmed, committed_ts_comfirmed, 
                                   local_committed_ts_comfirmed) == true) {
      break;
    }
  }
}
template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_rebuild_index_with_multiple_thread (
                                 Table<StaticConfig>   *tblp,
                                 BTreeIndexUniqueU64   *idx,
                                 Context<StaticConfig> *ctx,
                                 uint64_t              &committed_ts_comfirmed,
                                 std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs,
                                 uint16_t thread_id,
                                 uint16_t thread_cnt) {
  uint64_t local_committed_ts_comfirmed = committed_ts_comfirmed;
  for (uint16_t thd_id=0; thd_id< num_threads_; thd_id++) {
    auto tbl = reinterpret_cast<PmemTable<StaticConfig>*>(tblp);
    auto avfct = tbl->aep_rvs_[StaticConfig::kPKeyColumnHead][thd_id];  // pointer
    auto dt_sz = tbl->data_size_hint(StaticConfig::kPKeyColumnHead);
    auto rv_sz = dt_sz+sizeof(PmemRowVersion<StaticConfig>);

    size_t pg_thd = avfct->pages.size()/thread_cnt;
    size_t pg_start  = pg_thd*thread_id;
    size_t pg_end = pg_start+pg_thd;
    if (thread_id+1 == thread_cnt) {
      pg_end = avfct->pages.size();
    }

    for (auto pg_id=pg_start; pg_id< pg_end; pg_id++) {
      char *p = avfct->pages[pg_id];
      char *pend = p+PagePool<StaticConfig>::kPageSize/rv_sz*rv_sz;
      for ( ;p< pend; p+= rv_sz) {
        PmemRowVersion<StaticConfig> *rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>(p);
        // compare committed t2 comfirmed
        if (rv->cmark == 1) {
          committed_ts_comfirmed = std::max(committed_ts_comfirmed, rv->wts.t2);
        }
        if (rv->wts.t2<= committed_ts_comfirmed) {
          // committed rv, lookup in index
          uint64_t pkey = rv->pkey;
          uint64_t result[16], pos=0;
          auto func = [&result, &pos] (auto &k, auto &v) {
            (void)k;
            result[pos++] = v;
            return false;
          };
          uint64_t ret_num = idx->non_transactional_lookup_ptr(pkey, func);

          // ret_num decide to insert or update
          if (ret_num == 0) {
            uint64_t val = reinterpret_cast<uint64_t>(rv) | StaticConfig::kRowIsInPmemDevice;
            idx->non_transactional_insert (ctx, pkey, val);
          }
          else {
#if defined (DEBUG)
            assert (ret_num == 1 && pos == 1);
#endif
            // this logical row with pkey, has inserted a version
            for (uint64_t k=0; k< pos; k++) {
              uint64_t idx_rv_u64 = *reinterpret_cast<uint64_t*>(result[k]);
              auto idx_rv = reinterpret_cast<PmemRowVersion<StaticConfig>*>
                            (idx_rv_u64 & StaticConfig::kRowIsInPmemMask);
              if (rv->wts.t2 < idx_rv->wts.t2) {
                // rv is an older version, garbage it, maybe garbage to a random thread
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, ctx->thread_id(), rv);
                }
              }
              else {
                // rv is a newer version, idx points to it, garbage idx_rv
                if (StaticConfig::kRecoveryWithGarbage) {
                  tbl->aep_free_row_version(StaticConfig::kPKeyColumnHead, 
                                            ctx->thread_id(), idx_rv);
                }
                uint64_t val=reinterpret_cast<uint64_t>(idx_rv)|StaticConfig::kRowIsInPmemDevice;
                // modify idx
                *reinterpret_cast<uint64_t*>(result[k]) = val;
              }
            }
          }
        }
        else {
          // pending rv
          pending_rvs.push_back (rv);
        }
      }
    } // end of page loop
  } // end of thread loop

  while (local_committed_ts_comfirmed > committed_ts_comfirmed) {
    if (__sync_bool_compare_and_swap (&committed_ts_comfirmed, committed_ts_comfirmed, 
                                   local_committed_ts_comfirmed) == true) {
      break;
    }
  }
}

template <class StaticConfig>
void DB<StaticConfig>::recovery_zen_with_multiple_thread (uint16_t thread_cnt) {

  // multiple thread access one row with little chance, decide by comparing current timestamp
  // whether enough to atomic modify timestamp of row
  // if index search for the same row, seems need spin-lock because little chance of contention
  // compare_and_swap for spin-lock test_and_free for unlock
  // not clear
  // safe way, may not best, deal with different region, maybe in cache, read all
  // not parallel enough
  // now i think spin-lock for each primary key is best, it's spin-lock

  struct timeval net_t1, net_t2, net_t3, net_t4, net_t5, net_t6;
  gettimeofday (&net_t1, nullptr);

  Context<StaticConfig> *ctx = context();
  std::unordered_map<Table<StaticConfig>*, IndexStruct> 
    tbl_2_new_idx;
  std::unordered_map<Table<StaticConfig>*, std::vector<PmemRowVersion<StaticConfig>*>> 
    tbl_2_pending_aep[StaticConfig::kMaxLCoreCount];

  size_t expected_pending_num = 
    StaticConfig::kTimestampCheckpointPerContext*num_threads_/thread_cnt;

  for (auto i= index_mp_2_struct_ps.begin(); i!= index_mp_2_struct_ps.end(); i++) {
    auto tbl = i->first;

    for (uint16_t j=0; j< thread_cnt; j++) {
      tbl_2_pending_aep[j][tbl].reserve(expected_pending_num);
    }

    auto isp = i->second;    
    auto isps= isp.primary_index;

    assert (isps.type == TypeHashIndexUniqueU64 || isps.type ==TypeBtreeIndexUniqueU64);

    if (isps.type == TypeHashIndexUniqueU64) {
      uint64_t expected_row_count = isps.pointer.hash_unique_u64->expected_num_rows();

      // free original 
      // isps.pointer.hash_unique_u64->idx_tbl_->~Table<StaticConfig>();

      const uint64_t kDataSizes[] = {HashIndexUniqueU64::kDataSize};
      auto idx = new HashIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true),
                 expected_row_count);

      isps.pointer.hash_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    else if (isps.type == TypeBtreeIndexUniqueU64) {
      const uint64_t kDataSizes[] = {BTreeIndexUniqueU64::kDataSize};

      // free original 
      // isps.pointer.btree_unique_u64->idx_tbl_->~Table<StaticConfig>();

      auto idx = new BTreeIndexUniqueU64(
                 this, tbl, new Table<StaticConfig>(this, 1, kDataSizes, true));

      isps.pointer.btree_unique_u64 = idx;
      Transaction<StaticConfig> tx(ctx);
      idx->init (&tx);
    }
    tbl_2_new_idx[tbl] = isps;
  }

  // DEBUG
  printf ("recovery_zen: create new index\n");
  fflush (stdout);

  gettimeofday (&net_t2, nullptr);
  // get currentlly comfirmed committed timestamp
  uint64_t committed_ts_comfirmed = ts_ckp_.get_maximum_timestamp ();
  gettimeofday (&net_t3, nullptr);

  // DEBUG
  printf ("recovery_zen: committed_ts_comfirmed\n");

  auto func = [&] (uint16_t thd_id) {
    Context<StaticConfig> *ctx = ctxs_[thd_id];
    ::mica::util::lcore.pin_thread(thd_id);
 
    // rebuild primary index based on committed timestamp
    for (auto i=tbl_2_pending_aep[thd_id].begin(); i!= tbl_2_pending_aep[thd_id].end(); i++) {
      auto tbl  = i->first;            // table pointer
      std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs = i->second;
      auto isps = tbl_2_new_idx[tbl];  // hash or btree index pointer
  
      if (isps.type == TypeHashIndexUniqueU64) {  // decrease overhead
        auto idx = isps.pointer.hash_unique_u64;
        recovery_zen_rebuild_index_with_multiple_thread 
            (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs, thd_id, thread_cnt);
      } // end of index type hash
      else if (isps.type == TypeBtreeIndexUniqueU64) {  // decrease overhead
        auto idx = isps.pointer.btree_unique_u64;
        recovery_zen_rebuild_index_with_multiple_thread 
            (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs, thd_id, thread_cnt);
      } // end of type==TypeBtreeIndex
    } // end of for tbl_2_pending_aep
  };

  gettimeofday (&net_t4, nullptr);  
  std::vector<std::thread> threads;
  for (uint16_t i=1; i<thread_cnt; i++) {
    threads.emplace_back (func, i);
  }
  func (0);
  while (threads.size() > 0) {
    threads.back().join();
    threads.pop_back();
  }
  gettimeofday (&net_t5, nullptr);
 
  // DEBUG
  printf ("recovery_zen: rebuild index\n");
  // because the number of pending is small, multi-thread may have overhead
  for (uint16_t j=0; j< thread_cnt; j++) {
    // rebuild with pending rv based on current comfirmed t2 
    for (auto i=tbl_2_pending_aep[j].begin(); i!= tbl_2_pending_aep[j].end(); i++) {
      auto tbl  = i->first;            // table pointer
      std::vector<PmemRowVersion<StaticConfig>*> &pending_rvs = i->second;
      auto isps = tbl_2_new_idx[tbl];  // hash or btree index pointer
  
      if (isps.type == TypeHashIndexUniqueU64) {       // decrease overhead
        auto idx = isps.pointer.hash_unique_u64;
        recovery_zen_rebuild_pending_index
          (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
      } // end of index type hash
      else if (isps.type == TypeBtreeIndexUniqueU64) { // decrease overhead
        auto idx = isps.pointer.btree_unique_u64;
        recovery_zen_rebuild_pending_index
          (tbl, idx, ctx, committed_ts_comfirmed, pending_rvs);
      } // end of type==TypeBtreeIndex
    }   // end of for tbl_2_pending_aep
  }
  gettimeofday (&net_t6, nullptr);

  uint64_t net_use_1 = (uint64_t)((net_t3.tv_sec-net_t2.tv_sec)*1000000+(net_t3.tv_usec-net_t2.tv_usec));
  uint64_t net_use_2 = (uint64_t)((net_t5.tv_sec-net_t4.tv_sec)*1000000+(net_t5.tv_usec-net_t4.tv_usec));
  uint64_t net_use_3 = (uint64_t)((net_t6.tv_sec-net_t5.tv_sec)*1000000+(net_t6.tv_usec-net_t5.tv_usec));
 
  uint64_t net_time_use = net_use_1 + net_use_2 + net_use_3;

  printf ("net time use: %7.3lf ms\n", static_cast<double>(net_time_use)/1000.0);
  printf ("-- zen recover idntify ts:       %7.3lf ms\n", 
              static_cast<double>(net_use_1)/1000.0);
  printf ("-- zen recover main index:       %7.3lf ms\n", 
              static_cast<double>(net_use_2)/1000.0);
  printf ("-- zen recover pending versions: %7.3lf ms\n", 
              static_cast<double>(net_use_3)/1000.0);
  uint64_t trivial_use = (uint64_t)((net_t2.tv_sec-net_t1.tv_sec)*1000000+(net_t2.tv_usec-net_t1.tv_usec));
  printf ("trivial time use: %7.3lf ms\n", static_cast<double>(trivial_use)/1000.0);

  // DEBUG
  // printf ("recovery_zen: rebuild pending\n");
}

#endif  // end of CZEN multiple recovery
#endif  // end of RECOVERY

} // end of transaction
} // end of mica
#endif
