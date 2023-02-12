#pragma once

#include "global.h"
#include "helper.h"
#include "index_base.h"

class row_t;
class txn_man;

//TODO make proper variables private
// each BucketNode contains items sharing the same key
class BucketNode {
 public:
  BucketNode(idx_key_t key) { init(key); };
  void init(idx_key_t key) {
    this->key = key;
    next = NULL;
    items = NULL;
  }
  idx_key_t key;
  // The node for the next key
  BucketNode* next;
  // NOTE. The items can be a list of items connected by the next pointer.
  itemid_t* items;
};

// BucketHeader does concurrency control of Hash
class BucketHeader {
 public:
  void init();
  void insert_item(idx_key_t key, itemid_t* item, int part_id);
  void read_item(idx_key_t key, itemid_t*& item);
  BucketNode* first_node;
  uint64_t node_cnt;
  bool locked;
};

// TODO Hash index does not support partition yet.
class IndexHash : public index_base {
 public:
  RC init(uint64_t part_cnt, uint64_t bucket_cnt);
  RC init(uint64_t part_cnt, table_t* table, uint64_t bucket_cnt);

  RC index_insert(txn_man* txn, idx_key_t key, row_t* row, int part_id);
  RC index_remove(txn_man* txn, idx_key_t key, row_t*, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }

  RC index_read(txn_man* txn, idx_key_t key, row_t** row, int part_id);
  RC index_read_multiple(txn_man* txn, idx_key_t key, row_t** rows, size_t& count,
                         int part_id);

  RC index_read_range(txn_man* txn, idx_key_t min_key, idx_key_t max_key, row_t** rows,
                      size_t& count, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }
  RC index_read_range_rev(txn_man* txn, idx_key_t min_key, idx_key_t max_key, row_t** rows,
                          size_t& count, int part_id) {
    // Not implemented.
    assert(false);
    return ERROR;
  }

 private:
  void get_latch(BucketHeader* bucket);
  void release_latch(BucketHeader* bucket);

  // TODO implement more complex hash function
  uint64_t hash(idx_key_t key) { return key % _bucket_cnt_per_part; }

  BucketHeader** _buckets;
  uint64_t _bucket_cnt;
  uint64_t _bucket_cnt_per_part;
};
