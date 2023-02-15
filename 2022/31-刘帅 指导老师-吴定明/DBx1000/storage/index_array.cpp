#include "global.h"
#include "index_array.h"
#include "mem_alloc.h"
#include "table.h"

RC IndexArray::init(uint64_t part_cnt, uint64_t bucket_cnt) {
#if CC_ALG == MICA
  assert(false);
#endif

  (void)part_cnt;

  locked = false;
  size = bucket_cnt;
  arr = (row_t**)mem_allocator.alloc(sizeof(row_t*) * size, 0);
  for (size_t i = 0; i < size; i++) arr[i] = reinterpret_cast<row_t*>(-1);
  return RCOK;
}

RC IndexArray::init(uint64_t part_cnt, table_t* table, uint64_t bucket_cnt) {
  (void)part_cnt;

  this->table = table;

  locked = false;
  size = bucket_cnt;
  arr = (row_t**)mem_allocator.alloc(sizeof(row_t*) * size, 0);
  for (size_t i = 0; i < size; i++) arr[i] = reinterpret_cast<row_t*>(-1);
  return RCOK;
}

void IndexArray::get_latch() {
  while (!ATOM_CAS(locked, false, true)) PAUSE;
}

void IndexArray::release_latch() {
  bool ok = ATOM_CAS(locked, true, false);
  assert(ok);
  (void)ok;
}

RC IndexArray::index_insert(txn_man* txn, idx_key_t key, row_t* row, int part_id) {
  (void)part_id;

  get_latch();

  if (size <= key) {
    size_t new_size = (key + 1) * 2;
    auto new_arr = (row_t**)mem_allocator.alloc(sizeof(row_t*) * new_size, 0);
    memcpy(new_arr, arr, sizeof(row_t*) * size);
    for (size_t i = size; i < new_size; i++)
      new_arr[i] = reinterpret_cast<row_t*>(-1);

    mem_allocator.free(arr, sizeof(row_t*) * size);

    arr = new_arr;
    size = new_size;
  }

  arr[key] = row;

  release_latch();

  return RCOK;
}

RC IndexArray::index_read(txn_man* txn, idx_key_t key, row_t** row,
                          int part_id) {
  (void)txn;
  (void)part_id;

  if (key >= size) return ERROR;
  if (arr[key] == reinterpret_cast<row_t*>(-1)) return ERROR;
  *row = arr[key];
  return RCOK;
}
