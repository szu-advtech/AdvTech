#include <mm_malloc.h>
#include "global.h"
#include "table.h"
#include "catalog.h"
#include "row.h"
#include "txn.h"
#include "row_lock.h"
#include "row_ts.h"
#include "row_mvcc.h"
#include "row_hekaton.h"
#include "row_occ.h"
#include "row_tictoc.h"
#include "row_silo.h"
#include "row_vll.h"
#include "mem_alloc.h"
#include "manager.h"

#if defined(USE_INLINED_DATA) && (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE || CC_ALG == SILO || CC_ALG == TICTOC)

size_t row_t::alloc_size(table_t* t) { return sizeof(row_t) + t->get_schema()->get_tuple_size(); }

size_t row_t::max_alloc_size() { return sizeof(row_t) + MAX_TUPLE_SIZE; }

#else

size_t row_t::alloc_size(table_t* t) { return sizeof(row_t); }

size_t row_t::max_alloc_size() { return sizeof(row_t); }

#endif

RC
row_t::init(table_t * host_table, uint64_t part_id, uint64_t row_id) {
	_row_id = row_id;
	_part_id = part_id;
	this->table = host_table;
	is_deleted = 0;
  // debug
  // printf ("row_t::init\n");

#if CC_ALG == MICA
  // We ignore the given row_id argument to init() because it contains an
  // uninitialized value and is not used by the workload.

	assert(part_id >= 0 && part_id < table->mica_tbl.size());
  auto db = table->mica_db;
  auto tbl = table->mica_tbl[part_id];

  auto thread_id = ::mica::util::lcore.lcore_id()>>1; // get thread_id from cpu

  // debug
  // printf("lcore_id=%lu thread_id = %lu thread is_active=%d\n",
  //         ::mica::util::lcore.lcore_id(), thread_id, db->is_active(thread_id));

#if defined (test) || defined (CWBL)
  MICATransaction tx(db->context(thread_id));
#elif defined (CZEN)
  MICAPmemTransaction tx(db->context(thread_id));
#endif
        // db->activate(static_cast<uint16_t>(thread_id));
        // tx.aep_begin ();

	Catalog * schema = host_table->get_schema();
	//int tuple_size = schema->get_tuple_size();
	while (true) {

		if (!tx.begin())
			assert(false);

#if defined (test) || defined (CWBL)
#if !TPCC_CF
		MICAPmemRowAccessHandle rah(&tx);
		if (!rah.new_row(tbl, 0, MICATransaction::kNewRowID, false, schema->cf_sizes[0])) {
			if (!tx.abort())
				assert(false);
			continue;
		}
		_row_id = rah.row_id();
                data = rah.data();
#else
		MICARowAccessHandle rah(&tx);
                _row_id = MICATransaction::kNewRowID;
                for (uint64_t cf_id = 0; cf_id < table->get_schema()->cf_count; cf_id++) {
                  if (!rah.new_row(tbl, cf_id, _row_id, false, schema->cf_sizes[cf_id])) {
			if (!tx.abort())
				assert(false);
			continue;
                  }
                  if (cf_id == 0)
                    _row_id = rah.row_id();
                  cf_data[cf_id] = rah.data();
                  rah.reset();
                }
#endif
	  if (!tx.commit())
			continue;

		break;
#elif defined (CZEN)
#if !TPCC_CF
		MICAPmemRowAccessHandle rah(&tx);
		if (!rah.aep_new_row(tbl, 0, MICATransaction::kNewRowID, row_id, false, schema->cf_sizes[0])) {
                        // debug
                        // printf ("storage/row.cpp error aep_new_row!\n");

			if (!tx.aep_abort())
				assert(false);
			continue;
		}
		_row_id = rah.row_id();
                data = rah.data();
#else
		MICAPmemRowAccessHandle rah(&tx);
                _row_id = MICATransaction::kNewRowID;

                for (uint64_t cf_id = 0; cf_id < table->get_schema()->cf_count; cf_id++) {
                  if (!rah.aep_new_row(tbl, cf_id, _row_id, row_id, false, schema->cf_sizes[cf_id])) {
			if (!tx.aep_abort())
				assert(false);
			continue;
                  }
                  if (cf_id == 0)
                    _row_id = rah.row_id();
                  cf_data[cf_id] = rah.data();
                  rah.reset();
                }
#endif
	  if (!tx.aep_commit()) {
                        // debug
                        // printf ("storage/row.cpp aep_commit error\n");
			continue;
          }

		break;
#endif
	}
#else
#if defined(USE_INLINED_DATA) && (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE || CC_ALG == SILO || CC_ALG == TICTOC)
	// We can just use &data[0].
#else
	Catalog * schema = host_table->get_schema();
	int tuple_size = schema->get_tuple_size();
	data = (char *) mem_allocator.alloc(sizeof(char) * tuple_size, part_id);
#endif
#endif
	return RCOK;
}
void
row_t::init(int size)
{
#if CC_ALG != MICA
#if defined(USE_INLINED_DATA) && (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE || CC_ALG == SILO || CC_ALG == TICTOC)
	// We can just use &data[0].
#else
	data = (char *) mem_allocator.alloc(size, 64);
#endif
#else
	assert(false);
#endif
  is_deleted = 0;
}

RC
row_t::switch_schema(table_t * host_table) {
#if CC_ALG != MICA
	this->table = host_table;
#else
	assert(false);
#endif
	return RCOK;
}

void row_t::init_manager(row_t * row) {
#if CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE
#ifdef USE_INLINED_DATA
	// We can just use &manager[0].
#else
    manager = (Row_lock *) mem_allocator.alloc(sizeof(Row_lock), _part_id);
#endif
#elif CC_ALG == TIMESTAMP
    manager = (Row_ts *) mem_allocator.alloc(sizeof(Row_ts), _part_id);
#elif CC_ALG == MVCC
    manager = (Row_mvcc *) mem_allocator.alloc(sizeof(Row_mvcc), _part_id);
#elif CC_ALG == HEKATON
    manager = (Row_hekaton *) mem_allocator.alloc(sizeof(Row_hekaton), _part_id);
#elif CC_ALG == OCC
    manager = (Row_occ *) mem_allocator.alloc(sizeof(Row_occ), _part_id);
#elif CC_ALG == TICTOC
#ifdef USE_INLINED_DATA
	// We can just use &manager[0].
#else
	manager = (Row_tictoc *) mem_allocator.alloc(sizeof(Row_tictoc), _part_id);
#endif
#elif CC_ALG == SILO
#ifdef USE_INLINED_DATA
	// We can just use &manager[0].
#else
	manager = (Row_silo *) mem_allocator.alloc(sizeof(Row_silo), _part_id);
#endif
#elif CC_ALG == VLL
    manager = (Row_vll *) mem_allocator.alloc(sizeof(Row_vll), _part_id);
#endif

#if CC_ALG != HSTORE && CC_ALG != MICA
	manager->init(this);
#endif
}

table_t * row_t::get_table() {
	return table;
}

Catalog * row_t::get_schema() {
	return get_table()->get_schema();
}

const char * row_t::get_table_name() {
	return get_table()->get_table_name();
};
uint64_t row_t::get_tuple_size() {
	return get_schema()->get_tuple_size();
}

uint64_t row_t::get_field_cnt() {
	return get_schema()->field_cnt;
}

void row_t::set_value(int id, void * ptr) {
#if !TPCC_CF
	int datasize = get_schema()->get_field_size(id);
	int pos = get_schema()->get_field_index(id);
	memcpy( &data[pos], ptr, datasize);
#else
	int datasize = get_schema()->get_field_size(id);
	int pos = get_schema()->get_field_index(id);
        int cf_id = get_schema()->get_field_cf_id(id);
	memcpy( &cf_data[cf_id][pos], ptr, datasize);
#endif
}

void row_t::set_value(int id, void * ptr, int size) {
#if !TPCC_CF
	int pos = get_schema()->get_field_index(id);
	memcpy( &data[pos], ptr, size);
#else
	int pos = get_schema()->get_field_index(id);
        int cf_id = get_schema()->get_field_cf_id(id);
	memcpy( &cf_data[cf_id][pos], ptr, size);
#endif
}

void row_t::set_value(const char * col_name, void * ptr) {
	uint64_t id = get_schema()->get_field_id(col_name);
	set_value(id, ptr);
}

SET_VALUE(uint64_t);
SET_VALUE(int64_t);
SET_VALUE(double);
SET_VALUE(UInt32);
SET_VALUE(SInt32);
SET_VALUE(UInt16);
SET_VALUE(SInt16);
SET_VALUE(UInt8);
SET_VALUE(SInt8);

GET_VALUE(uint64_t);
GET_VALUE(int64_t);
GET_VALUE(double);
GET_VALUE(UInt32);
GET_VALUE(SInt32);
GET_VALUE(UInt16);
GET_VALUE(SInt16);
GET_VALUE(UInt8);
GET_VALUE(SInt8);


char * row_t::get_value(int id) {
#if !TPCC_CF
	int pos = get_schema()->get_field_index(id);
	return &data[pos];
#else
	int pos = get_schema()->get_field_index(id);
        int cf_id = get_schema()->get_field_cf_id(id);
	return &cf_data[cf_id][pos];
#endif
}

char * row_t::get_value(char * col_name) {
#if !TPCC_CF
	uint64_t pos = get_schema()->get_field_index(col_name);
	return &data[pos];
#else
	int id = get_schema()->get_field_id(col_name);
        return get_value(id);
#endif
}

#if !TPCC_CF
char * row_t::get_data() { return data; }

void row_t::set_data(char * data, uint64_t size) {
#if CC_ALG == MICA
	printf("oops\n");
	assert(false);
#endif
	memcpy(this->data, data, size);
}
#endif

// copy from the src to this
void row_t::copy(row_t * src) {
#if CC_ALG == MICA
	printf("oops\n");
	assert(false);
#else
	set_data(src->get_data(), src->get_tuple_size());
#endif
}

void row_t::free_row() {
#if CC_ALG != MICA
#if defined(USE_INLINED_DATA) && (CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE || CC_ALG == SILO || CC_ALG == TICTOC)
#else
	free(data);
#endif
#endif
}

RC row_t::get_row(access_t type, txn_man * txn, row_t *& row) {
#if CC_ALG == MICA
	printf("deprecated\n");
	assert(false);
	return ERROR;
#else
	// Not supported in other algorithms.
	assert(type != PEEK && type != SKIP);

	RC rc = RCOK;
#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
	uint64_t thd_id = txn->get_thd_id();
	lock_t lt = (type == RD || type == SCAN)? LOCK_SH : LOCK_EX;
#if CC_ALG == DL_DETECT
	uint64_t * txnids;
	int txncnt;
	rc = this->manager->lock_get(lt, txn, txnids, txncnt);
#else
	rc = this->manager->lock_get(lt, txn);
#endif

	if (rc == RCOK) {
		row = this;
	} else if (rc == Abort) {}
	else if (rc == WAIT) {
		ASSERT(CC_ALG == WAIT_DIE || CC_ALG == DL_DETECT);
		uint64_t starttime = get_sys_clock();
#if CC_ALG == DL_DETECT
		bool dep_added = false;
#endif
		uint64_t endtime;
		txn->lock_abort = false;
		INC_STATS(txn->get_thd_id(), wait_cnt, 1);
		while (!txn->lock_ready && !txn->lock_abort)
		{
#if CC_ALG == WAIT_DIE
			continue;
#elif CC_ALG == DL_DETECT
			uint64_t last_detect = starttime;
			uint64_t last_try = starttime;

			uint64_t now = get_sys_clock();
			if (now - starttime > g_timeout ) {
				txn->lock_abort = true;
				break;
			}
			if (g_no_dl) {
				PAUSE
				continue;
			}
			int ok = 0;
			if ((now - last_detect > g_dl_loop_detect) && (now - last_try > DL_LOOP_TRIAL)) {
				if (!dep_added) {
					ok = dl_detector.add_dep(txn->get_txn_id(), txnids, txncnt, txn->row_cnt);
					if (ok == 0)
						dep_added = true;
					else if (ok == 16)
						last_try = now;
				}
				if (dep_added) {
					ok = dl_detector.detect_cycle(txn->get_txn_id());
					if (ok == 16)  // failed to lock the deadlock detector
						last_try = now;
					else if (ok == 0)
						last_detect = now;
					else if (ok == 1) {
						last_detect = now;
					}
				}
			} else
				PAUSE
#endif
		}
		if (txn->lock_ready)
			rc = RCOK;
		else if (txn->lock_abort) {
			rc = Abort;
			return_row(type, txn, NULL);
		}
		endtime = get_sys_clock();
		INC_TMP_STATS(thd_id, time_wait, endtime - starttime);
		row = this;
	}
	return rc;
#elif CC_ALG == TIMESTAMP || CC_ALG == MVCC || CC_ALG == HEKATON
	uint64_t thd_id = txn->get_thd_id();
	// For TIMESTAMP RD, a new copy of the row will be returned.
	// for MVCC RD, the version will be returned instead of a copy
	// So for MVCC RD-WR, the version should be explicitly copied.
	//row_t * newr = NULL;
  #if CC_ALG == TIMESTAMP
	// TODO. should not call malloc for each row read. Only need to call malloc once
	// before simulation starts, like TicToc and Silo.
	txn->cur_row = (row_t *) mem_allocator.alloc(alloc_size(get_table()), this->get_part_id());
	txn->cur_row->init(get_table(), this->get_part_id());
  #endif

	// TODO need to initialize the table/catalog information.
	TsType ts_type = (type == RD)? R_REQ : P_REQ;
	rc = this->manager->access(txn, ts_type, row);
	if (rc == RCOK ) {
		row = txn->cur_row;
	} else if (rc == WAIT) {
		uint64_t t1 = get_sys_clock();
		while (!txn->ts_ready)
			PAUSE
		uint64_t t2 = get_sys_clock();
		INC_TMP_STATS(thd_id, time_wait, t2 - t1);
		row = txn->cur_row;
	}
	if (rc != Abort) {
		row->table = get_table();
		assert(row->get_schema() == this->get_schema());
	}
	return rc;
#elif CC_ALG == OCC
	// OCC always make a local copy regardless of read or write
	txn->cur_row = (row_t *) mem_allocator.alloc(alloc_size(get_table()), get_part_id());
	txn->cur_row->init(get_table(), get_part_id());
	rc = this->manager->access(txn, R_REQ);
	row = txn->cur_row;
	return rc;
#elif CC_ALG == TICTOC || CC_ALG == SILO
	// like OCC, tictoc also makes a local copy for each read/write
	row->table = get_table();
	TsType ts_type = (type == RD)? R_REQ : P_REQ;
	rc = this->manager->access(txn, ts_type, row);
	return rc;
#elif CC_ALG == HSTORE || CC_ALG == VLL
	row = this;
	return rc;
#else
	assert(false);
#endif
#endif
}

#if CC_ALG == MICA
#if !TPCC_CF
RC row_t::get_row(access_t type, txn_man* txn, table_t* table,
                  row_t* access_row, uint64_t row_id, uint64_t part_id) {
#else
RC row_t::get_row(access_t type, txn_man* txn, table_t* table,
                  row_t* access_row, uint64_t row_id, uint64_t part_id,
                  const access_t* cf_access_type) {
#endif
  assert(part_id >= 0 && part_id < table->mica_tbl.size());

  // printf("get_row row_id=%lu row_count=%lu\n", item->row_id,
  //        table->mica_tbl[this->get_part_id()]->row_count());

  access_row->table = table;
  access_row->set_part_id(part_id);
  access_row->set_row_id(row_id);

#if !TPCC_CF
    access_t this_type = type;
    uint64_t cf_id = 0;
#else
  for (uint64_t cf_id = 0; cf_id < table->get_schema()->cf_count; cf_id++) {
    access_t this_type = cf_access_type != NULL ? cf_access_type[cf_id] : type;
#endif

#if TPCC_CF
    if (this_type == SKIP) {
      access_row->cf_data[cf_id] = NULL;
      continue;
    }
#endif

    if (this_type == PEEK) {

#if defined (test) || defined (CWBL)
      MICARowAccessHandlePeekOnly rah(txn->mica_tx);
      if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, false,
                        false))
#elif defined (CZEN)
      MICAPmemRowAccessHandlePeekOnly rah(txn->mica_tx);
      if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false, false,
                        false))
#endif
        return Abort;

#if !TPCC_CF
      access_row->data = const_cast<char*>(rah.cdata());
#else
      access_row->cf_data[cf_id] = const_cast<char*>(rah.cdata());
#endif
    } else {

#if defined (test) || defined (CWBL)
      MICARowAccessHandle rah(txn->mica_tx);
      if (this_type == RD) {
        if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, true,
                          false) ||
            !rah.read_row())
          return Abort;
      } else if (this_type == WR) {
        if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, true,
                          true) ||
            !rah.read_row() || !rah.write_row())
          return Abort;
      } else {
        assert(false);
        return Abort;
      }
#elif defined (CZEN)
      MICAPmemRowAccessHandle rah(txn->mica_tx);
      if (this_type == RD) {
        if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false/*false default*/, true,
                          false) ||
            !rah.aep_read_row()) {

          // debug
          // printf ("storage/row.cpp-1: aep_peek_row/aep_read_row error! row_id=%p\n", (void*)row_id);
          return Abort;
        }
      } else if (this_type == WR) {
        if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false/*false default*/, true,
                          true) ||
            !rah.aep_read_row() || !rah.aep_write_row(static_cast<uint64_t>(-1))) {

          // debug
          // printf ("storage/row.cpp-2: aep_peek_row/aep_read_row error! row_id=%p\n", (void*)row_id);
          return Abort;
        }
      } else {
        assert(false);
        return Abort;
      }
#endif

#if !TPCC_CF
      if (this_type == RD)
        access_row->data = const_cast<char*>(rah.cdata());
      else
        access_row->data = rah.data();
#else
      if (this_type == RD)
        access_row->cf_data[cf_id] = const_cast<char*>(rah.cdata());
      else
        access_row->cf_data[cf_id] = rah.data();
#endif
    }

#if TPCC_CF
  }
#endif
  return RCOK;
}
#endif

// the "row" is the row read out in get_row().
// For locking based CC_ALG, the "row" is the same as "this".
// For timestamp based CC_ALG, the "row" != "this", and the "row" must be freed.
// For MVCC, the row will simply serve as a version. The version will be
// delete during history cleanup.
// For TIMESTAMP, the row will be explicity deleted at the end of access().
// (cf. row_ts.cpp)
void row_t::return_row(access_t type, txn_man * txn, row_t * row) {
#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
	assert (row == NULL || row == this || type == XP);
	if (ROLL_BACK && type == XP) {// recover from previous writes.
		this->copy(row);
	}
	this->manager->lock_release(txn);
#elif CC_ALG == TIMESTAMP || CC_ALG == MVCC
	// for RD or SCAN or XP, the row should be deleted.
	// because all WR should be companied by a RD
	// for MVCC RD, the row is not copied, so no need to free.
  #if CC_ALG == TIMESTAMP
	if (type == RD || type == SCAN) {
		row->free_row();
		mem_allocator.free(row, alloc_size(get_table()));
	}
  #endif
	if (type == XP) {
		this->manager->access(txn, XP_REQ, row);
	} else if (type == WR) {
		assert (type == WR && row != NULL);
		assert (row->get_schema() == this->get_schema());
		RC rc = this->manager->access(txn, W_REQ, row);
		assert(rc == RCOK);
	}
#elif CC_ALG == OCC
	assert (row != NULL);
	if (type == WR)
		manager->write( row, txn->end_ts );
	row->free_row();
	mem_allocator.free(row, alloc_size(get_table()));
	return;
#elif CC_ALG == TICTOC || CC_ALG == SILO
	assert (row != NULL);
	return;
#elif CC_ALG == HSTORE || CC_ALG == VLL
	return;
#elif CC_ALG == MICA
	return;
#else
	assert(false);
#endif
}
