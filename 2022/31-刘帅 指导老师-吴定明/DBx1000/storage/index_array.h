#pragma once

#include "global.h"
#include "helper.h"
#include "index_base.h"

class row_t;
class txn_man;

class IndexArray : public index_base {
 public:
  RC init(uint64_t part_cnt, uint64_t bucket_cnt);
  RC init(uint64_t part_cnt, table_t* table, uint64_t bucket_cnt);

  RC index_insert(txn_man* txn, idx_key_t key, row_t* row, int part_id);
  RC index_remove(txn_man* txn, idx_key_t key, row_t* row, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }

  RC index_read(txn_man* txn, idx_key_t key, row_t** row, int part_id);
  RC index_read_multiple(txn_man* txn, idx_key_t key, row_t** rows,
                         size_t& count, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }

  RC index_read_range(txn_man* txn, idx_key_t min_key, idx_key_t max_key,
                      row_t** rows, size_t& count, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }
  RC index_read_range_rev(txn_man* txn, idx_key_t min_key, idx_key_t max_key,
                          row_t** rows, size_t& count, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }

 private:
  void get_latch();
  void release_latch();

  bool locked;
  row_t** arr;
  size_t size;
};
