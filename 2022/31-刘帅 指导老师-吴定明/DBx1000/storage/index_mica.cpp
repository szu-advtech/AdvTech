#include "global.h"
#include "index_mica.h"
#include "mem_alloc.h"
#include "table.h"
#include "row.h"
#include "helper.h"
#include "txn.h"

#if INDEX_STRUCT == IDX_MICA

using ::mica::transaction::BTreeRangeType;

template <typename MICAIndexT>
RC IndexMICAGeneric<MICAIndexT>::init(uint64_t part_cnt, table_t* table) {
  if (typeid(MICAIndexT) == typeid(MICAOrderedIndex))
    return init(part_cnt, table, 0);
  else {
    assert(false);
    return ERROR;
  }
}

template <>
RC IndexMICAGeneric<MICAIndex>::init(uint64_t part_cnt, table_t* table,
                                     uint64_t bucket_cnt) {
  this->table = table;
  this->bucket_cnt = bucket_cnt;

  auto db = table->mica_db;

  bucket_cnt = (bucket_cnt + part_cnt - 1) / part_cnt;

  // printf("idx=%p part_cnt=%d bucket_cnt=%" PRIu64 "\n", this, part_cnt,
  // bucket_cnt);

  // debug
  // printf ("index_mica.cpp part_cnt = %lu\n", part_cnt);
  // return RCOK;

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    uint64_t thread_id = part_id % g_thread_cnt;
    ::mica::util::lcore.pin_thread(thread_id);

    auto mica_tbl = table->mica_tbl[part_id];

    char buf[1024];
    int i = 0;
    while (true) {
      sprintf(buf, "%s_IDX_%d", table->get_table_name(), i);
      if (mica_tbl->db()->create_hash_index_nonunique_u64(buf, mica_tbl,
                                                          bucket_cnt))
        break;
      i++;
    }
    auto p = mica_tbl->db()->get_hash_index_nonunique_u64(buf);
    assert(p != nullptr);

    MICATransaction tx(db->context(thread_id));
    if (!p->init(&tx)) {
      assert(false);
      return ERROR;
    }

    printf("idx_name=%s part_id=%" PRIu64 " part_cnt=%" PRIu64
           " bucket_cnt=%" PRIu64 "\n",
           buf, part_id, part_cnt, bucket_cnt);

    mica_idx.push_back(p);
  }

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::init(uint64_t part_cnt, table_t* table,
                                            uint64_t bucket_cnt) {
  this->table = table;
  this->bucket_cnt = bucket_cnt;

  auto db = table->mica_db;

  bucket_cnt = (bucket_cnt + part_cnt - 1) / part_cnt;

  // printf("idx=%p part_cnt=%d bucket_cnt=%" PRIu64 "\n", this, part_cnt,
  // bucket_cnt);

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    uint64_t thread_id = part_id % g_thread_cnt;
    ::mica::util::lcore.pin_thread(thread_id);

    auto mica_tbl = table->mica_tbl[part_id];

    char buf[1024];
    int i = 0;
    while (true) {
      sprintf(buf, "%s_IDX_%d", table->get_table_name(), i);
      // if (mica_tbl->db()->create_btree_index_nonunique_u64(buf, mica_tbl))
      if (mica_tbl->db()->create_btree_index_unique_u64(buf, mica_tbl)) break;
      i++;
    }
    // auto p = mica_tbl->db()->get_btree_index_nonunique_u64(buf);
    auto p = mica_tbl->db()->get_btree_index_unique_u64(buf);
    assert(p != nullptr);

    MICATransaction tx(db->context(thread_id));
    if (!p->init(&tx)) {
      assert(false);
      return ERROR;
    }

    printf("idx_name=%s part_id=%" PRIu64 " part_cnt=%" PRIu64
           " bucket_cnt=%" PRIu64 "\n",
           buf, part_id, part_cnt, bucket_cnt);

    mica_idx.push_back(p);
  }

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_insert(txn_man* txn, MICATransaction* tx,
                                             idx_key_t key, row_t* row,
                                             int part_id) {
  auto ret = mica_idx[part_id]->insert(tx, key, (uint64_t)row);
  if (ret == MICAIndex::kHaveToAbort) return Abort;
  if (ret != 1) return ERROR;

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_remove(txn_man* txn, MICATransaction* tx,
                                             idx_key_t key, row_t* row,
                                             int part_id) {
  auto ret = mica_idx[part_id]->remove(tx, key, (uint64_t)row);
  if (ret == MICAIndex::kHaveToAbort) return Abort;
  if (ret != 1) return ERROR;

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_read(txn_man* txn, idx_key_t key,
                                           row_t** row, int part_id) {
  auto tx = txn->mica_tx;
  bool skip_validation = !(MICA_FULLINDEX);

  auto ret = mica_idx[part_id]->lookup(tx, key, skip_validation,
                                       [&row](auto& key, auto value) {
                                         *row = (row_t*)value;
                                         return false;
                                       });
  if (ret == 0) {
    // debug
    // printf ("index_mica.cpp:index_read ERROE ret==0\n");
    // fflush (stdout);
    return ERROR;
  }
  if (ret == MICAIndex::kHaveToAbort) {
    // debug
    // printf ("index_mica.cpp:index_read Abort ret==kHaveToAbort\n");
    // fflush (stdout);
    return Abort;
  }

  // debug
  // printf("index_mica.cpp:index_read key=%lu ret=%lu val=%p\n", key, ret, (void*)row);
  // fflush (stdout);
 
  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_read_multiple(txn_man* txn, idx_key_t key,
                                                    row_t** rows, size_t& count,
                                                    int part_id) {
  auto tx = txn->mica_tx;
  bool skip_validation = !(MICA_FULLINDEX);

  uint64_t i = 0;
  uint64_t ret = mica_idx[part_id]->lookup(tx, key, skip_validation,
                                           [&i, rows, count](auto& k, auto v) {
                                             rows[i++] = (row_t*)v;
                                             // printf("%" PRIu64 "\n", i);
                                             return i < count;
                                           });
  if (ret == MICAIndex::kHaveToAbort) return Abort;
  count = i;
  // printf("%lu %lu\n", key, (uint64_t)row);
  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_read_range(txn_man* txn,
                                                 idx_key_t min_key,
                                                 idx_key_t max_key,
                                                 row_t** rows, size_t& count,
                                                 int part_id) {
  // Not supported.
  assert(false);
  return ERROR;
}

template <>
RC IndexMICAGeneric<MICAIndex>::index_read_range_rev(
    txn_man* txn, idx_key_t min_key, idx_key_t max_key, row_t** rows,
    size_t& count, int part_id) {
  // Not supported.
  assert(false);
  return ERROR;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_insert(txn_man* txn,
                                                    MICATransaction* tx,
                                                    idx_key_t key, row_t* row,
                                                    int part_id) {
  auto ret = mica_idx[part_id]->insert(tx, key, (uint64_t)row);
  if (ret == MICAIndex::kHaveToAbort) return Abort;
  if (ret != 1) return ERROR;

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_remove(txn_man* txn,
                                                    MICATransaction* tx,
                                                    idx_key_t key, row_t* row,
                                                    int part_id) {
  auto ret = mica_idx[part_id]->remove(tx, key, (uint64_t)row);
  if (ret == MICAIndex::kHaveToAbort) return Abort;
  if (ret != 1) return ERROR;

  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_read(txn_man* txn, idx_key_t key,
                                                  row_t** row, int part_id) {
  auto tx = txn->mica_tx;
#if SIMPLE_INDEX_UPDATE || !TPCC_VALIDATE_NODE
  bool skip_validation = !(MICA_FULLINDEX);
#else
  bool skip_validation = false;
#endif

  // auto ret =
  //     mica_idx[part_id]
  //         ->lookup<BTreeRangeType::kInclusive, BTreeRangeType::kOpen, false>(
  //             tx, std::make_pair(key, 0), std::make_pair(key, 0),
  //             skip_validation, [&key, &row](auto& k, auto v) {
  //               if (k.first == key) *row = (row_t*)k.second;
  //               return false;
  //             });
  auto ret = mica_idx[part_id]->lookup(tx, key, skip_validation,
                                       [&key, &row](auto& k, auto v) {
                                         *row = (row_t*)v;
                                         return false;
                                       });
  if (ret == 0) return ERROR;
  if (ret == MICAOrderedIndex::kHaveToAbort) return Abort;
  // printf("%lu %lu\n", key, (uint64_t)row);
  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_read_multiple(
    txn_man* txn, idx_key_t key, row_t** rows, size_t& count, int part_id) {
  auto tx = txn->mica_tx;
#if SIMPLE_INDEX_UPDATE || !TPCC_VALIDATE_NODE
  bool skip_validation = !(MICA_FULLINDEX);
#else
  bool skip_validation = false;
#endif

  uint64_t i = 0;
  // uint64_t ret =
  //     mica_idx[part_id]
  //         ->lookup<BTreeRangeType::kInclusive, BTreeRangeType::kOpen, false>(
  //             tx, std::make_pair(key, 0), std::make_pair(key, 0),
  //             skip_validation, [&key, &i, rows, count](auto& k, auto v) {
  //               if (k.first == key) {
  //                 rows[i++] = (row_t*)k.second;
  //                 return i < count;
  //               } else
  //                 return false;
  //             });
  uint64_t ret = mica_idx[part_id]->lookup(
      tx, key, skip_validation, [&key, &i, rows, count](auto& k, auto v) {
        rows[i++] = (row_t*)v;
        return i < count;
      });
  if (ret == MICAOrderedIndex::kHaveToAbort) return Abort;
  count = i;
  // printf("%lu %lu\n", key, (uint64_t)rows[0]);
  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_read_range(
    txn_man* txn, idx_key_t min_key, idx_key_t max_key, row_t** rows,
    size_t& count, int part_id) {
  auto tx = txn->mica_tx;
#if SIMPLE_INDEX_UPDATE || !TPCC_VALIDATE_NODE
  bool skip_validation = !(MICA_FULLINDEX);
#else
  bool skip_validation = false;
#endif

  uint64_t i = 0;
  // uint64_t ret = mica_idx[part_id]
  //                    ->lookup<BTreeRangeType::kInclusive,
  //                             BTreeRangeType::kInclusive, false>(
  //                        tx, std::make_pair(min_key, 0),
  //                        std::make_pair(max_key, uint64_t(-1)), skip_validation,
  //                        [&i, rows, count](auto& k, auto v) {
  //                          rows[i++] = (row_t*)k.second;
  //                          return i < count;
  //                        });
  uint64_t ret =
      mica_idx[part_id]
          ->lookup<BTreeRangeType::kInclusive, BTreeRangeType::kInclusive,
                   false>(tx, min_key, max_key, skip_validation,
                          [&i, rows, count](auto& k, auto v) {
                            rows[i++] = (row_t*)v;
                            return i < count;
                          });
  if (ret == MICAOrderedIndex::kHaveToAbort) return Abort;
  count = i;
  // printf("%lu %lu\n", key, (uint64_t)rows[0]);
  return RCOK;
}

template <>
RC IndexMICAGeneric<MICAOrderedIndex>::index_read_range_rev(
    txn_man* txn, idx_key_t min_key, idx_key_t max_key, row_t** rows,
    size_t& count, int part_id) {
  auto tx = txn->mica_tx;
#if SIMPLE_INDEX_UPDATE || !TPCC_VALIDATE_NODE
  bool skip_validation = !(MICA_FULLINDEX);
#else
  bool skip_validation = false;
#endif

  uint64_t i = 0;
  // uint64_t ret = mica_idx[part_id]
  //                    ->lookup<BTreeRangeType::kInclusive,
  //                             BTreeRangeType::kInclusive, true>(
  //                        tx, std::make_pair(min_key, 0),
  //                        std::make_pair(max_key, uint64_t(-1)), skip_validation,
  //                        [&i, rows, count](auto& k, auto v) {
  //                          rows[i++] = (row_t*)k.second;
  //                          return i < count;
  //                        });
  uint64_t ret =
      mica_idx[part_id]
          ->lookup<BTreeRangeType::kInclusive, BTreeRangeType::kInclusive,
                   true>(tx, min_key, max_key, skip_validation,
                         [&i, rows, count](auto& k, auto v) {
                           rows[i++] = (row_t*)v;
                           return i < count;
                         });
  if (ret == MICAOrderedIndex::kHaveToAbort) return Abort;
  count = i;
  // printf("%lu %lu\n", key, (uint64_t)rows[0]);
  return RCOK;
}

template class IndexMICAGeneric<MICAIndex>;

template class IndexMICAGeneric<MICAOrderedIndex>;

#endif
