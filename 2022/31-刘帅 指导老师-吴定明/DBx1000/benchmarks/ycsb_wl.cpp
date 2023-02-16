#include <sched.h>
#include "global.h"
#include "helper.h"
#include "ycsb.h"
#include "wl.h"
#include "thread.h"
#include "table.h"
#include "row.h"
#include "index_hash.h"
#include "index_btree.h"
#include "index_mbtree.h"
#include "index_mica.h"
#include "index_mica_mbtree.h"
#include "catalog.h"
#include "manager.h"
#include "row_lock.h"
#include "row_ts.h"
#include "row_mvcc.h"
#include "mem_alloc.h"
#include "query.h"
#include <random>

int ycsb_wl::next_tid;

RC ycsb_wl::init() {
  workload::init();
  next_tid = 0;
  char* cpath = getenv("GRAPHITE_HOME");
  string path;
  if (cpath == NULL)
    path = "./benchmarks/YCSB_schema.txt";
  else {
    path = string(cpath);
    path += "/tests/apps/dbms/YCSB_schema.txt";
  }
  // return RCOK;

  init_schema(path);

  // return RCOK;

  // Randomize the data layout by shuffling row insert order.
  std::mt19937 g;
  g.seed(std::random_device()());
  shuffled_ids.reserve(g_synth_table_size);
  for (uint64_t i = 0; i < g_synth_table_size; i++)
    shuffled_ids.push_back(i);
  std::shuffle(shuffled_ids.begin(), shuffled_ids.end(), g);

  // return RCOK;

  init_table_parallel();
  //	init_table();
  return RCOK;
}

RC ycsb_wl::init_schema(string schema_file) {
  workload::init_schema(schema_file);
  the_table = tables["MAIN_TABLE"];
  the_index = hash_indexes["HASH_MAIN_INDEX"];
  return RCOK;
}

int ycsb_wl::key_to_part(uint64_t key) {
  uint64_t rows_per_part = g_synth_table_size / g_part_cnt;
  return key / rows_per_part;
}

/*
RC ycsb_wl::init_table() {
  RC rc;
  uint64_t total_row = 0;
  while (true) {
    for (UInt32 part_id = 0; part_id < g_part_cnt; part_id++) {
      if (total_row > g_synth_table_size) goto ins_done;
      row_t* new_row = NULL;
#if CC_ALG == MICA
      row_t row_container;
      new_row = &row_container;
#endif
      uint64_t row_id;
      rc = the_table->get_new_row(new_row, part_id, row_id);
      // TODO insertion of last row may fail after the table_size
      // is updated. So never access the last record in a table
      assert(rc == RCOK);
      uint64_t primary_key = shuffled_ids[total_row];
      new_row->set_primary_key(primary_key);
      new_row->set_value(0, &primary_key);
      Catalog* schema = the_table->get_schema();
      for (UInt32 fid = 0; fid < schema->get_field_cnt(); fid++) {
        int field_size = schema->get_field_size(fid);
        char value[field_size];
        for (int i = 0; i < field_size; i++) value[i] = (char)rand() % (1 << 8);
        new_row->set_value(fid, value);
      }
      uint64_t idx_key = primary_key;
      index_insert(the_index, idx_key, new_row, part_id);
      total_row++;
    }
  }
ins_done:
  printf("[YCSB] Table \"MAIN_TABLE\" initialized.\n");
  return RCOK;
}
*/

// init table in parallel
void ycsb_wl::init_table_parallel() {
  assert(g_init_parallelism <= g_thread_cnt);

  enable_thread_mem_pool = true;
  pthread_t p_thds[g_init_parallelism - 1];
  for (UInt32 i = 0; i < g_init_parallelism - 1; i++)
    pthread_create(&p_thds[i], NULL, threadInitTable, this);
  threadInitTable(this);

  for (uint32_t i = 0; i < g_init_parallelism - 1; i++) {
    int rc = pthread_join(p_thds[i], NULL);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
  }
  enable_thread_mem_pool = false;
  mem_allocator.unregister();
}

void* ycsb_wl::init_table_slice() {
  UInt32 tid = ATOM_FETCH_ADD(next_tid, 1);
// set cpu affinity
#if CC_ALG == MICA
  ::mica::util::lcore.pin_thread(tid);
  // debug
  // printf("tid=%u g_thread_cnt=%u lcore_id=%lu\n", tid, g_thread_cnt,
  //        ::mica::util::lcore.lcore_id());

  mica_db->activate(static_cast<uint16_t>(tid));
  // debug
  // printf ("1-thread_id=%u is_active = %d\n", tid, mica_db->is_active(tid));
#else
  set_affinity(tid);
#endif

  mem_allocator.register_thread(tid);

  RC rc;
  assert(tid < g_init_parallelism);
  while ((UInt32)ATOM_FETCH_ADD(next_tid, 0) < g_init_parallelism) {
  }
  assert((UInt32)ATOM_FETCH_ADD(next_tid, 0) == g_init_parallelism);
  uint64_t slice_size = (g_synth_table_size + g_init_parallelism) / g_init_parallelism;
  uint64_t slice_offset = slice_size * tid;
  if (slice_offset + slice_size >= g_synth_table_size)
    slice_size = g_synth_table_size - slice_offset;

    // debug
    // printf ("2-thread_id=%u is_active = %d\n", tid, mica_db->is_active(tid));

  for (uint64_t key_i = slice_offset; key_i < slice_offset + slice_size; key_i++) {
    uint64_t key = shuffled_ids[key_i];
    row_t* new_row = NULL;
#if CC_ALG == MICA
    row_t row_container;
    new_row = &row_container;
#endif

#if defined (test) || defined (CWBL)
    uint64_t row_id;
#elif defined (CZEN)
    uint64_t row_id = key; // CZEN
#endif

    int part_id = key_to_part(key);
    rc = the_table->get_new_row(new_row, part_id, row_id);
    assert(rc == RCOK);
    uint64_t primary_key = key;
    new_row->set_primary_key(primary_key);
    new_row->set_value(0, &primary_key);
    Catalog* schema = the_table->get_schema();

    for (UInt32 fid = 0; fid < schema->get_field_cnt(); fid++) {
      char value[6] = "hello";
      new_row->set_value(fid, value);
    }

    uint64_t idx_key = primary_key;

    // debug
    // printf ("3-thread_id=%u is_active = %d\n", tid, mica_db->is_active(tid));

    index_insert(the_index, idx_key, new_row, part_id);

/*  // debug
#if defined (test) || defined (CWBL)
    index_insert(the_index, idx_key, new_row, part_id);
#elif defined (CZEN)
    index_insert(the_index, idx_key, (row_t*)(new_row->get_row_id()), part_id);
#endif
*/

    // 		if (key % 1000000 == 0) {
    // 			printf("key=%" PRIu64 "\n", key);
    // #if INDEX_STRUCT == IDX_MICA
    // 			uint64_t k = 0;
    // 			the_table->mica_tbl->renew_rows(mica_db->context(static_cast<uint16_t>(tid)), k, static_cast<uint64_t>(-1), false);
    // 			for (auto idx : the_index->mica_idx) {
    // 				auto mica_tbl = idx->index_table();
    // 				k = 0;
    // 				mica_tbl->renew_rows(mica_db->context(static_cast<uint16_t>(tid)), k, static_cast<uint64_t>(-1), false);
    // 			}
    // #endif
    //		}
  }

#if CC_ALG == MICA
  mica_db->deactivate(static_cast<uint16_t>(tid));
#endif
  return NULL;
}

RC ycsb_wl::get_txn_man(txn_man*& txn_manager, thread_t* h_thd) {
  txn_manager = (ycsb_txn_man*)mem_allocator.alloc(sizeof(ycsb_txn_man), h_thd->get_thd_id());
  new (txn_manager) ycsb_txn_man();
  txn_manager->init(h_thd, this, h_thd->get_thd_id());
  return RCOK;
}
