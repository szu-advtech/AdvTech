#pragma once

#include "global.h"
#include "helper.h"
#include "index_base.h"

#if INDEX_STRUCT == IDX_MICA

class row_t;
class table_t;
class itemid_t;
class txn_man;

class IndexMICAMBTree : public index_base {
 public:
  RC init(uint64_t part_cnt, table_t* table);
  RC init(uint64_t part_cnt, table_t* table, uint64_t bucket_cnt);

  RC index_insert(txn_man* txn, MICATransaction* tx, idx_key_t key, row_t* row,
                  int part_id);
  // This method ignores the second row_t* argument.
  RC index_remove(txn_man* txn, MICATransaction* tx, idx_key_t key, row_t*,
                  int part_id);

  RC index_read(txn_man* txn, idx_key_t key, row_t** row, int part_id);
  RC index_read_multiple(txn_man* txn, idx_key_t key, row_t** rows,
                         size_t& count, int part_id);

  RC index_read_range(txn_man* txn, idx_key_t min_key, idx_key_t max_key,
                      row_t** rows, size_t& count, int part_id);
  RC index_read_range_rev(txn_man* txn, idx_key_t min_key, idx_key_t max_key,
                          row_t** rows, size_t& count, int part_id);

  static RC validate(txn_man* txn);

#if TPCC_VALIDATE_GAP
  RC list_init(MICADB* mica_db, uint64_t gap_off);
  RC list_make(MICADB* mica_db);
  RC list_insert(MICATransaction* tx, idx_key_t key, row_t* row, int part_id);
  RC list_remove(MICATransaction* tx, idx_key_t key, row_t* row, int part_id);
#endif

  table_t* table;
  std::vector<void*> btree_idx;

#if TPCC_VALIDATE_GAP
  bool validate_gap;
  uint64_t gap_off;

  uint64_t neg_infty;
  uint64_t pos_infty;
#endif
};

#endif
