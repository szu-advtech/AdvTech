#define CONFIG_H "silo/config/config-perf.h"
#include "silo/rcu.h"
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "txn.h"
#include "row.h"
#include "wl.h"
#include "ycsb.h"
#include "thread.h"
#include "mem_alloc.h"
#include "occ.h"
#include "table.h"
#include "catalog.h"
#include "index_btree.h"
#include "index_hash.h"
#include "index_array.h"
#include "index_mbtree.h"
#include "index_mica.h"
#include "index_mica_mbtree.h"

void txn_man::init(thread_t * h_thd, workload * h_wl, uint64_t thd_id) {
	this->h_thd = h_thd;
	this->h_wl = h_wl;
	pthread_mutex_init(&txn_lock, NULL);
	lock_ready = false;
	ready_part = 0;
	row_cnt = 0;
	wr_cnt = 0;
	insert_cnt = 0;
	remove_cnt = 0;
	insert_idx_cnt = 0;
	remove_idx_cnt = 0;
  node_map.clear();
	accesses = (Access **) mem_allocator.alloc(sizeof(Access *) * MAX_ROW_PER_TXN, thd_id);
	for (int i = 0; i < MAX_ROW_PER_TXN; i++)
		accesses[i] = NULL;
	num_accesses_alloc = 0;
#if CC_ALG == TICTOC || CC_ALG == SILO
	_pre_abort = (g_params["pre_abort"] == "true");
	if (g_params["validation_lock"] == "no-wait")
		_validation_no_wait = true;
	else if (g_params["validation_lock"] == "waiting")
		_validation_no_wait = false;
	else
		assert(false);
#endif
#if CC_ALG == TICTOC
	_max_wts = 0;
	_write_copy_ptr = (g_params["write_copy_form"] == "ptr");
	_atomic_timestamp = (g_params["atomic_timestamp"] == "true");
#elif CC_ALG == SILO
	_cur_tid = 0;
#elif CC_ALG == MICA
	// printf("thd_id=%" PRIu64 "\n", thd_id);

#if defined (test) || defined (CWBL)
	mica_tx = new MICATransaction(h_wl->mica_db->context(thd_id));
#elif defined (CZEN)
	mica_tx = new MICAPmemTransaction(h_wl->mica_db->context(thd_id));
#endif

#endif

}

void txn_man::set_txn_id(txnid_t txn_id) {
	this->txn_id = txn_id;
}

txnid_t txn_man::get_txn_id() {
	return this->txn_id;
}

workload * txn_man::get_wl() {
	return h_wl;
}

uint64_t txn_man::get_thd_id() {
	return h_thd->get_thd_id();
}

void txn_man::set_ts(ts_t timestamp) {
	this->timestamp = timestamp;
}

ts_t txn_man::get_ts() {
	return this->timestamp;
}

RC txn_man::apply_index_changes(RC rc) {
#if RCU_ALLOC
  assert(rcu::s_instance.in_rcu_region());
#endif

#if INDEX_STRUCT != IDX_MICA || (INDEX_STRUCT == IDX_MICA && defined(IDX_MICA_USE_MBTREE))

#if !SIMPLE_INDEX_UPDATE
  if (rc == RCOK) rc = ORDERED_INDEX::validate(this);

  if (rc != RCOK) {
    // Remove previously inserted placeholders.
    for (size_t i = 0; i < insert_idx_cnt; i++) {
      auto idx = insert_idx_idx[i];
      auto key = insert_idx_key[i];
      // auto row = insert_idx_row[i];
      auto part_id = insert_idx_part_id[i];
#if INDEX_STRUCT != IDX_MICA
      auto rc_remove = idx->index_remove(this, key, NULL, part_id);
#else
      auto rc_remove = idx->index_remove(this, mica_tx, key, NULL, part_id);
#endif
      assert(rc_remove == RCOK);
    }
    insert_idx_cnt = 0;
    return rc;
  }
#else
  if (rc != RCOK) return rc;
#endif  // SIMPLE_INDEX_UPDATE

#if SIMPLE_INDEX_UPDATE
  for (size_t i = 0; i < insert_idx_cnt; i++) {
    auto idx = insert_idx_idx[i];
    auto key = insert_idx_key[i];
    auto row = insert_idx_row[i];
    auto part_id = insert_idx_part_id[i];

    // printf("insert_idx idx=%p key=%" PRIu64 " part_id=%d\n", idx, key, part_id);
#if INDEX_STRUCT != IDX_MICA
    auto rc_insert = idx->index_insert(this, key, row, part_id);
#else
    auto rc_insert = idx->index_insert(this, mica_tx, key, row, part_id);
#endif

    if (rc_insert != RCOK) {
      // Remove previously inserted entries.
      while (i > 0) {
        i--;
        auto idx = insert_idx_idx[i];
        auto key = insert_idx_key[i];
        // auto row = insert_idx_row[i];
        auto part_id = insert_idx_part_id[i];
#if INDEX_STRUCT != IDX_MICA
        auto rc_remove = idx->index_remove(this, key, NULL, part_id);
#else
        auto rc_remove = idx->index_remove(this, mica_tx, key, NULL, part_id);
#endif
        assert(rc_remove == RCOK);
      }
      insert_idx_cnt = 0;
      return Abort;
    }
  }
#endif  // SIMPLE_INDEX_UPDATE
  insert_idx_cnt = 0;

	for (size_t i = 0; i < remove_idx_cnt; i++) {
		auto idx = remove_idx_idx[i];
		auto key = remove_idx_key[i];
		auto part_id = remove_idx_part_id[i];
    // printf("remove_idx idx=%p key=%" PRIu64 " part_id=%d\n", idx, key, part_id);

#if INDEX_STRUCT != IDX_MICA
		auto rc_remove = idx->index_remove(this, key, NULL, part_id);
#else
		auto rc_remove = idx->index_remove(this, mica_tx, key, NULL, part_id);
#endif
		assert(rc_remove == RCOK);
	}
	remove_idx_cnt = 0;
#endif

#if CC_ALG != MICA
	// Free deleted rows
	for (size_t i = 0; i < remove_cnt; i++) {
		auto row = remove_rows[i];
		assert(!row->is_deleted);
		row->is_deleted = 1;
		// printf("remove_row row_id=%" PRIu64 " part_id=%" PRIu64 "\n", row->get_row_id(), row->get_part_id());
		// XXX: Freeing the row immediately is unsafe due to concurrent access.
		// We do this only when using RCU.
	  if (RCU_ALLOC) mem_allocator.free(row, row_t::alloc_size(row->get_table()));
		// XXX: We need to perform the following to free up all the resources
// #if CC_ALG != HSTORE && CC_ALG != OCC && CC_ALG != MICA && !defined(USE_INLINED_DATA)
// 			// XXX: Need to find the manager size.
// 			mem_allocator.free(row->manager, 0);
// #endif
// 			row->free_row();
	}
	remove_cnt = 0;
#endif

	return rc;
}

void txn_man::cleanup(RC rc) {
#if CC_ALG == HEKATON || CC_ALG == MICA
#if CC_ALG == HEKATON
	if (rc == Abort) {
		for (UInt32 i = 0; i < insert_cnt; i ++) {
			row_t * row = insert_rows[i];
      row->is_deleted = 1;
      row->manager->release();

#if CC_ALG != HSTORE && CC_ALG != OCC && CC_ALG != MICA && !defined(USE_INLINED_DATA)
			// XXX: Need to find the manager size.
			mem_allocator.free(row->manager, 0);
#endif
      // We cannot free data for Hekaton because of pending reads.
			//row->free_row();
			mem_allocator.free(row, row_t::alloc_size(row->get_table()));
    }
  }
#endif

	row_cnt = 0;
	wr_cnt = 0;
	insert_cnt = 0;
	remove_cnt = 0;
	insert_idx_cnt = 0;
	remove_idx_cnt = 0;
  node_map.clear();
	return;

#else

	for (int rid = row_cnt - 1; rid >= 0; rid --) {
		row_t * orig_r = accesses[rid]->orig_row;
		access_t type = accesses[rid]->type;
		if (type == WR && rc == Abort)
			type = XP;

#if (CC_ALG == NO_WAIT || CC_ALG == DL_DETECT) && ISOLATION_LEVEL == REPEATABLE_READ
		if (type == RD) {
			accesses[rid]->data = NULL;
			continue;
		}
#endif

		if (ROLL_BACK && type == XP &&
					(CC_ALG == DL_DETECT ||
					CC_ALG == NO_WAIT ||
					CC_ALG == WAIT_DIE))
		{
			orig_r->return_row(type, this, accesses[rid]->orig_data);
		} else {
			orig_r->return_row(type, this, accesses[rid]->data);
		}
#if CC_ALG != TICTOC && CC_ALG != SILO && CC_ALG != MICA
		accesses[rid]->data = NULL;
#endif
	}

	if (rc == Abort) {
		for (UInt32 i = 0; i < insert_cnt; i ++) {
			row_t * row = insert_rows[i];
      row->is_deleted = 1;

#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
      auto rc = row->manager->lock_release(this);
      assert(rc == RCOK);
#elif CC_ALG == TICTOC || CC_ALG == SILO
      row->manager->release();
#elif CC_ALG == HEKATON
      // This is handled above.
      // row->manager->release();
      assert(false);
#else
      // Not implemented.
      assert(false);
#endif

#if CC_ALG != HSTORE && CC_ALG != OCC && CC_ALG != MICA && !defined(USE_INLINED_DATA)
			// XXX: Need to find the manager size.
			mem_allocator.free(row->manager, 0);
#endif
			row->free_row();
			mem_allocator.free(row, row_t::alloc_size(row->get_table()));
		}
	} else {
		for (UInt32 i = 0; i < insert_cnt; i ++) {
			row_t * row = insert_rows[i];
#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
      auto rc = row->manager->lock_release(this);
      assert(rc == RCOK);
#elif CC_ALG == TICTOC || CC_ALG == SILO
      // Unlocking new rows is done in validate_*() to initialize row TID.
      (void)row;
#elif CC_ALG == HEKATON
      // Unlocking new rows is done in validate_*() to initialize row TID.
      (void)row;
#else
      // Not implemented.
      assert(false);
#endif
    }
  }
	row_cnt = 0;
	wr_cnt = 0;
	insert_cnt = 0;
	remove_cnt = 0;
	insert_idx_cnt = 0;
	remove_idx_cnt = 0;
  node_map.clear();
#if CC_ALG == DL_DETECT
	dl_detector.clear_dep(get_txn_id());
#endif
#endif
}

// index_read methods
template <typename IndexT>
RC
txn_man::index_read(IndexT* index, idx_key_t key, row_t** row, int part_id) {
	return index->index_read(this, key, row, part_id);
}

template <typename IndexT>
RC
txn_man::index_read_multiple(IndexT* index, idx_key_t key, row_t** rows, size_t& count, int part_id) {
	return index->index_read_multiple(this, key, rows, count, part_id);
}

template <typename IndexT>
RC
txn_man::index_read_range(IndexT* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id) {
	return index->index_read_range(this, min_key, max_key, rows, count, part_id);
}

template <typename IndexT>
RC
txn_man::index_read_range_rev(IndexT* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id) {
	return index->index_read_range_rev(this, min_key, max_key, rows, count, part_id);
}

// get_row methods
#if CC_ALG != MICA
template <typename IndexT>
#if !TPCC_CF
row_t* txn_man::get_row(IndexT* index, row_t* row, int part_id, access_t type) {
#else
row_t* txn_man::get_row(IndexT* index, row_t* row, int part_id, access_t type, const access_t* cf_access_type) {
#endif
	(void)index;
	(void)part_id;
#if TPCC_CF
        assert(cf_access_type == NULL);
#endif

	if (type == PEEK)
		return row;

	if (CC_ALG == HSTORE)
		return row;

	// uint64_t starttime = get_sys_clock();
	RC rc = RCOK;
	if (accesses[row_cnt] == NULL) {
		Access * access = (Access *) mem_allocator.alloc(sizeof(Access), -1);
		accesses[row_cnt] = access;
#if (CC_ALG == SILO || CC_ALG == TICTOC)
		access->data = (row_t *) mem_allocator.alloc(row_t::max_alloc_size(), -1);
		access->data->init(MAX_TUPLE_SIZE);
		access->orig_data = (row_t *) mem_allocator.alloc(row_t::max_alloc_size(), -1);
		access->orig_data->init(MAX_TUPLE_SIZE);
#elif (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE)
		access->orig_data = (row_t *) mem_allocator.alloc(row_t::max_alloc_size(), -1);
		access->orig_data->init(MAX_TUPLE_SIZE);
#endif
		num_accesses_alloc ++;
	}

	// Initial deleted row detection to reduce creating a new local row.
	if (row->is_deleted)
		return NULL;

	rc = row->get_row(type, this, accesses[ row_cnt ]->data);

	if (rc == Abort) {
		return NULL;
	}

	// Check if the original row is deleted after getting the local row.
	// This avoids a race condition so that we can simply use the version check for Silo/TicToc to detect any deletion perfomed by another thread.
	if (row->is_deleted)
		return NULL;

	accesses[row_cnt]->type = type;
	accesses[row_cnt]->orig_row = row;
#if CC_ALG == TICTOC
	accesses[row_cnt]->wts = last_wts;
	accesses[row_cnt]->rts = last_rts;
#elif CC_ALG == SILO
	accesses[row_cnt]->tid = last_tid;
#elif CC_ALG == HEKATON
	accesses[row_cnt]->history_entry = history_entry;
#endif

#if ROLL_BACK && (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE)
	if (type == WR) {
		accesses[row_cnt]->orig_data->table = row->get_table();
		accesses[row_cnt]->orig_data->copy(row);
	}
#endif

#if (CC_ALG == NO_WAIT || CC_ALG == DL_DETECT) && ISOLATION_LEVEL == REPEATABLE_READ
	if (type == RD)
		row->return_row(type, this, accesses[ row_cnt ]->data);
#endif

	row_cnt ++;
	if (type == WR)
		wr_cnt ++;

	// uint64_t timespan = get_sys_clock() - starttime;
	// INC_TMP_STATS(get_thd_id(), time_man, timespan);
	return accesses[row_cnt - 1]->data;
}
#else	// CC_ALG == MICA
#if !TPCC_CF
template <typename IndexT>
row_t *
txn_man::get_row(IndexT* index, row_t* row, int part_id, access_t type)
#else
template <typename IndexT>
row_t *
txn_man::get_row(IndexT* index, row_t* row, int part_id, access_t type, const access_t* cf_access_type)
#endif
{
	// printf("1 row_id=%lu\n", item->row_id);

	// uint64_t starttime = get_sys_clock();
	RC rc = RCOK;
	assert(row_cnt < MAX_ROW_PER_TXN);
	if (accesses[row_cnt] == NULL) {
		Access * access = (Access *) mem_allocator.alloc(sizeof(Access), -1);
		accesses[row_cnt] = access;
		access->data = (row_t *) mem_allocator.alloc(row_t::alloc_size(index->table), -1);
		num_accesses_alloc ++;
	}

	// printf("2 row_id=%lu\n", item->row_id);
#if !TPCC_CF
	rc = row_t::get_row(type, this, index->table, accesses[ row_cnt ]->data, (uint64_t)row, part_id);
#else
	rc = row_t::get_row(type, this, index->table, accesses[ row_cnt ]->data, (uint64_t)row, part_id, cf_access_type);
#endif
	// assert(rc == RCOK);

	if (rc == Abort) {
                // debug
                // printf ("system/txn.cpp: get_row error row=%p\n", row);

		return NULL;
	}

	row_cnt ++;
	// if (type == WR)
	//         wr_cnt ++;

	// uint64_t timespan = get_sys_clock() - starttime;
	// INC_TMP_STATS(get_thd_id(), time_man, timespan);
	return accesses[row_cnt - 1]->data;
}
#endif

// search
#if !TPCC_CF
template <typename IndexT>
row_t* txn_man::search(IndexT* index, uint64_t key, int part_id,
                        access_t type) {
	row_t* row;
  auto ret = index_read(index, key, &row, part_id);
	if (ret != RCOK) {

    // debug
    // printf ("system/txn.cpp:search index_read error\n");
    return NULL;
  }

  return get_row(index, row, part_id, type);
}
#else
template <typename IndexT>
row_t* txn_man::search(IndexT* index, uint64_t key, int part_id,
                        access_t type, const access_t* cf_access_type) {
	row_t* row;
  auto ret = index_read(index, key, &row, part_id);
	if (ret != RCOK) return NULL;

  return get_row(index, row, part_id, type, cf_access_type);
}
#endif

// insert_row/remove_row
bool txn_man::insert_row(table_t* tbl, row_t*& row, int part_id,
                          uint64_t& out_row_id) {
#if CC_ALG != MICA
  if (tbl->get_new_row(row, part_id, out_row_id) != RCOK) return false;
	assert(insert_cnt < MAX_ROW_PER_TXN);
	insert_rows[insert_cnt ++] = row;

#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
	auto rc = row->manager->lock_get(LOCK_EX, this);
  assert(rc == RCOK);
#elif CC_ALG == TICTOC || CC_ALG == SILO
	row->manager->lock();
#elif CC_ALG == HEKATON
	row->manager->lock();
#else
  // Not implemented.
  assert(false);
#endif

  return true;
#else
  assert(row != NULL);
  assert(part_id >= 0 && part_id < (int)tbl->mica_tbl.size());
#if !TPCC_CF
  MICARowAccessHandle rah(mica_tx);
  if (!rah.new_row(tbl->mica_tbl[part_id], 0, MICATransaction::kNewRowID, false, tbl->get_schema()->cf_sizes[0])) return false;
  out_row_id = rah.row_id();
  row->set_row_id(out_row_id);
  row->set_part_id(part_id);
  row->table = tbl;
  row->data = rah.data();
#else
  MICARowAccessHandle rah(mica_tx);
  out_row_id = MICATransaction::kNewRowID;
  for (uint64_t cf_id = 0; cf_id < tbl->get_schema()->cf_count; cf_id++) {
    if (!rah.new_row(tbl->mica_tbl[part_id], cf_id, out_row_id, false, tbl->get_schema()->cf_sizes[cf_id])) return false;
    if (cf_id == 0) {
      out_row_id = rah.row_id();
      row->set_row_id(out_row_id);
    }
    row->cf_data[cf_id] = rah.data();
    rah.reset();
  }
  row->set_part_id(part_id);
  row->table = tbl;
#endif
  return true;
#endif
}

bool txn_man::remove_row(row_t* row) {
#if CC_ALG != MICA
	remove_rows[remove_cnt++] = row;
  return true;
#else
  // MICA tables are directly managed.
	assert(false);
  return false;
#endif
}

// index_insert/index_remove
#if INDEX_STRUCT != IDX_MICA || defined(IDX_MICA_USE_MBTREE)
template <>
bool txn_man::insert_idx(ORDERED_INDEX* index, uint64_t key, row_t* row,
                            int part_id) {
#if CC_ALG == MICA
#if TPCC_VALIDATE_GAP
  if (index->list_insert(mica_tx, key, row, part_id) != RCOK)
    return false;
#endif

  row = (row_t*)row->get_row_id();
#endif

#if !SIMPLE_INDEX_UPDATE
#if INDEX_STRUCT != IDX_MICA
  auto rc_insert = index->index_insert(this, key, row, part_id);
#else
  auto rc_insert = index->index_insert(this, mica_tx, key, row, part_id);
#endif
  if (rc_insert != RCOK)
    return false;
#endif  // SIMPLE_INDEX_UPDATE

	assert(insert_idx_cnt < MAX_ROW_PER_TXN);
	insert_idx_idx[insert_idx_cnt] = index;
	insert_idx_key[insert_idx_cnt] = key;
	insert_idx_row[insert_idx_cnt] = row;
	insert_idx_part_id[insert_idx_cnt] = part_id;
	insert_idx_cnt++;
  return true;
}
#endif
#if INDEX_STRUCT == IDX_MICA
template <>
bool txn_man::insert_idx(OrderedIndexMICA* index, uint64_t key, row_t* row,
                            int part_id) {
  auto& mica_idx = index->mica_idx;
  // if (mica_idx[part_id]->insert(mica_tx, make_pair(key, row->row_id), 0) != 1)
  return mica_idx[part_id]->insert(mica_tx, key, row->get_row_id()) == 1;
}
#endif

#if INDEX_STRUCT != IDX_MICA || defined(IDX_MICA_USE_MBTREE)
template <>
bool txn_man::remove_idx(ORDERED_INDEX* index, uint64_t key, row_t* row,
                            int part_id) {
#if CC_ALG == MICA
#if TPCC_VALIDATE_GAP
  if (index->list_remove(mica_tx, key, row, part_id) != RCOK)
    return false;
#endif
#endif

  (void)row;
	assert(remove_idx_cnt < MAX_ROW_PER_TXN);
	remove_idx_idx[remove_idx_cnt] = index;
	remove_idx_key[remove_idx_cnt] = key;
	remove_idx_part_id[remove_idx_cnt] = part_id;
	remove_idx_cnt++;
  return true;
}
#endif
#if INDEX_STRUCT == IDX_MICA
template <>
bool txn_man::remove_idx(OrderedIndexMICA* index, uint64_t key, row_t* row, int part_id) {
  auto& mica_idx = index->mica_idx;
  // return mica_idx[part_id]->remove(mica_tx, make_pair(key, row_id), 0) == 1;
  return mica_idx[part_id]->remove(mica_tx, key, row->get_row_id()) == 1;
}
#endif

// template instantiation
template
RC txn_man::index_read(HASH_INDEX* index, idx_key_t key, row_t** row, int part_id);
template
RC txn_man::index_read_multiple(HASH_INDEX* index, idx_key_t key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range(HASH_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range_rev(HASH_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
#if !TPCC_CF
template
row_t* txn_man::get_row(HASH_INDEX* index, row_t* row, int part_id, access_t type);
template
row_t* txn_man::search(HASH_INDEX* index, size_t key, int part_id, access_t type);
#else
template
row_t* txn_man::get_row(HASH_INDEX* index, row_t* row, int part_id, access_t type, const access_t* cf_access_type);
template
row_t* txn_man::search(HASH_INDEX* index, size_t key, int part_id, access_t type, const access_t* cf_access_type);
#endif
// template
// bool txn_man::insert_idx(HASH_INDEX* idx, idx_key_t key, row_t* row, int part_id);
// template
// bool txn_man::remove_idx(HASH_INDEX* idx, idx_key_t key, row_t* row, int part_id);

template
RC txn_man::index_read(ARRAY_INDEX* index, idx_key_t key, row_t** row, int part_id);
template
RC txn_man::index_read_multiple(ARRAY_INDEX* index, idx_key_t key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range(ARRAY_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range_rev(ARRAY_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
#if !TPCC_CF
template
row_t* txn_man::get_row(ARRAY_INDEX* index, row_t* row, int part_id, access_t type);
template
row_t* txn_man::search(ARRAY_INDEX* index, size_t key, int part_id, access_t type);
#else
template
row_t* txn_man::get_row(ARRAY_INDEX* index, row_t* row, int part_id, access_t type, const access_t* cf_access_type);
template
row_t* txn_man::search(ARRAY_INDEX* index, size_t key, int part_id, access_t type, const access_t* cf_access_type);
#endif
// template
// bool txn_man::insert_idx(ARRAY_INDEX* idx, idx_key_t key, row_t* row, int part_id);
// template
// bool txn_man::remove_idx(ARRAY_INDEX* idx, idx_key_t key, row_t* row, int part_id);

template
RC txn_man::index_read(ORDERED_INDEX* index, idx_key_t key, row_t** row, int part_id);
template
RC txn_man::index_read_multiple(ORDERED_INDEX* index, idx_key_t key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range(ORDERED_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
template
RC txn_man::index_read_range_rev(ORDERED_INDEX* index, idx_key_t min_key, idx_key_t max_key, row_t** rows, size_t& count, int part_id);
#if !TPCC_CF
template
row_t* txn_man::get_row(ORDERED_INDEX* index, row_t* row, int part_id, access_t type);
template
row_t* txn_man::search(ORDERED_INDEX* index, size_t key, int part_id, access_t type);
#else
template
row_t* txn_man::get_row(ORDERED_INDEX* index, row_t* row, int part_id, access_t type, const access_t* cf_access_type);
template
row_t* txn_man::search(ORDERED_INDEX* index, size_t key, int part_id, access_t type, const access_t* cf_access_type);
#endif
// template
// bool txn_man::insert_idx(ORDERED_INDEX* idx, idx_key_t key, row_t* row, int part_id);
// template
// bool txn_man::remove_idx(ORDERED_INDEX* idx, idx_key_t key, row_t* row, int part_id);


RC txn_man::finish(RC rc) {
#if CC_ALG == HSTORE
	rc = apply_index_changes(rc);
	return rc;
#endif
	// uint64_t starttime = get_sys_clock();
#if CC_ALG == OCC
	if (rc == RCOK)
		rc = occ_man.validate(this);
	else
		cleanup(rc);
#elif CC_ALG == TICTOC
	if (rc == RCOK)
		rc = validate_tictoc();
	else
		cleanup(rc);
#elif CC_ALG == SILO
	if (rc == RCOK)
		rc = validate_silo();
	else
		cleanup(rc);
#elif CC_ALG == HEKATON
	rc = validate_hekaton(rc);
	cleanup(rc);
#elif CC_ALG == MICA

#if defined (test) || defined (CWBL)
  if (rc == RCOK) {
    if (mica_tx->has_began()) {
#ifndef IDX_MICA_USE_MBTREE
      rc = mica_tx->commit() ? RCOK : Abort;
#else
      auto write_func = [this]() { return apply_index_changes(RCOK) == RCOK; };
      rc = mica_tx->commit(NULL, write_func) ? RCOK : Abort;
      if (rc != RCOK) rc = apply_index_changes(rc);
#endif
    } else
      rc = RCOK;
  } else if (mica_tx->has_began() && !mica_tx->abort())
    assert(false);
  cleanup(rc);
#elif defined (CZEN)
  if (rc == RCOK) {
    if (mica_tx->has_began()) {
#ifndef IDX_MICA_USE_MBTREE
      rc = mica_tx->aep_commit() ? RCOK : Abort;
#else
      rc = mica_tx->aep_commit() ? RCOK : Abort;
      if (rc != RCOK) rc = apply_index_changes(rc);
#endif
    } else
      rc = RCOK;
  } else if (mica_tx->has_began() && !mica_tx->aep_abort())
    assert(false);
  cleanup(rc);
#endif
// end of CC_ALG ==MICA

#else
	rc = apply_index_changes(rc);
	cleanup(rc);
#endif

	// uint64_t timespan = get_sys_clock() - starttime;
	// INC_TMP_STATS(get_thd_id(), time_man,  timespan);
	// INC_STATS(get_thd_id(), time_cleanup,  timespan);
	return rc;
}

void
txn_man::release() {
	for (int i = 0; i < num_accesses_alloc; i++)
		mem_allocator.free(accesses[i], 0);
	mem_allocator.free(accesses, 0);
}
