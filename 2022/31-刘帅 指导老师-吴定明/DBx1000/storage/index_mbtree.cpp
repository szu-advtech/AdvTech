#define CONFIG_H "silo/config/config-perf.h"

#define NDB_MASSTREE 1
#include "silo/masstree/config.h"
#include "silo/masstree_btree.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "index_mbtree.h"
#include "helper.h"
#include "row.h"
#include "mem_alloc.h"
#include "txn.h"

// #ifdef DEBUG
// #undef NDEBUG
// #endif

struct mbtree_params : public Masstree::nodeparams<> {
  typedef row_t* value_type;
  typedef Masstree::value_print<value_type> value_print_type;
  typedef simple_threadinfo threadinfo_type;
  enum { RcuRespCaller = true };
};

typedef mbtree<mbtree_params> concurrent_mbtree;

class IndexMBTree_cb
    : public concurrent_mbtree::low_level_search_range_callback {
 public:
  IndexMBTree_cb(txn_man* txn, row_t** rows, uint64_t count, uint64_t& i)
      : txn_(txn), rows_(rows), count_(count), i_(i), abort_(false) {}

  void on_resp_node(const concurrent_mbtree::node_opaque_t* n,
                    uint64_t version) override {
#if TPCC_VALIDATE_NODE
    auto it = txn_->node_map.find((void*)n);
    if (it == txn_->node_map.end()) {
      // printf("index node seen: %p %" PRIu64 "\n", n, version);
      txn_->node_map.emplace_hint(it, (void*)n, version);
    } else if ((*it).second != version)
      abort_ = true;
#endif
  }

  bool invoke(const concurrent_mbtree::string_type& k,
              concurrent_mbtree::value_type v,
              const concurrent_mbtree::node_opaque_t* n,
              uint64_t version) override {
    (void)k;
    (void)n;
    (void)version;
    rows_[i_++] = v;
    return i_ < count_;
  }

  bool need_to_abort() const { return abort_; }

 private:
  txn_man* txn_;
  row_t** rows_;
  uint64_t count_;
  uint64_t& i_;
  bool abort_;
};

RC IndexMBTree::init(uint64_t part_cnt, table_t* table) {
  return init(part_cnt, table, 0);
}

RC IndexMBTree::init(uint64_t part_cnt, table_t* table, uint64_t bucket_cnt) {
  (void)bucket_cnt;

  this->table = table;

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    mem_allocator.register_thread(part_id % g_thread_cnt);

    auto t = (concurrent_mbtree*)mem_allocator.alloc(sizeof(concurrent_mbtree),
                                                     part_id);
    new (t) concurrent_mbtree;

    btree_idx.push_back(t);
  }

  return RCOK;
}

RC IndexMBTree::index_insert(txn_man* txn, idx_key_t key, row_t* row,
                             int part_id) {
  auto idx = reinterpret_cast<concurrent_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

#if !TPCC_VALIDATE_NODE
  if (!idx->insert_if_absent(mbtree_key, row)) return ERROR;
#else
  concurrent_mbtree::insert_info_t insert_info;
  if (!idx->insert_if_absent(mbtree_key, row, &insert_info)) return ERROR;

  // assert(concurrent_mbtree::ExtractVersionNumber(insert_info.node) ==
  //        insert_info.new_version);  // for single-threaded debugging

  if (txn) {
    auto it = txn->node_map.find((void*)insert_info.node);
    if (it == txn->node_map.end()) {
      // txn->node_map.emplace_hint(it, (void*)insert_info.node,
      //                            insert_info.new_version);
    } else if ((*it).second != insert_info.old_version) {
      // printf("index node version mismatch: %p previously seen: %" PRIu64
      //        " now: %" PRIu64 "\n",
      //        insert_info.node, (*it).second, insert_info.old_version);
      return Abort;
    } else {
      (*it).second = insert_info.new_version;
    }

    // printf("index node updated: %p old %" PRIu64 " new %" PRIu64 "\n",
    //        insert_info.node, insert_info.old_version, insert_info.new_version);
  }
#endif

  return RCOK;
}

RC IndexMBTree::index_read(txn_man* txn, idx_key_t key, row_t** row,
                           int part_id) {
  auto idx = reinterpret_cast<concurrent_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

  concurrent_mbtree::versioned_node_t search_info;
  if (!idx->search(mbtree_key, *row, &search_info)) {
#if TPCC_VALIDATE_NODE
    auto it = txn->node_map.find((void*)search_info.first);
    if (it == txn->node_map.end()) {
      txn->node_map.emplace_hint(it, (void*)search_info.first,
                                 search_info.second);
      // printf("index node seen: %p %" PRIu64 "\n", search_info.first,
      //        search_info.second);
    } else if ((*it).second != search_info.second)
      return Abort;
#endif
    return ERROR;
  }

  return RCOK;
}

RC IndexMBTree::index_read_multiple(txn_man* txn, idx_key_t key, row_t** rows,
                                    uint64_t& count, int part_id) {
  // Duplicate keys are currently not supported in IndexMBTree.
  assert(false);
  (void)txn;
  (void)key;
  (void)rows;
  (void)count;
  (void)part_id;
  return ERROR;
}

RC IndexMBTree::index_read_range(txn_man* txn, idx_key_t min_key,
                                 idx_key_t max_key, row_t** rows,
                                 uint64_t& count, int part_id) {
  if (count == 0) return RCOK;

  auto idx = reinterpret_cast<concurrent_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key_min(min_key);

  // mbtree's range is right-open.
  max_key++;
  assert(max_key != 0);
  u64_varkey mbtree_key_max(max_key);

  uint64_t i = 0;
  auto cb = IndexMBTree_cb(txn, rows, count, i);

  idx->search_range_call(mbtree_key_min, &mbtree_key_max, cb);
  if (cb.need_to_abort()) return Abort;

  count = i;

  return RCOK;
}

RC IndexMBTree::index_read_range_rev(txn_man* txn, idx_key_t min_key,
                                     idx_key_t max_key, row_t** rows,
                                     uint64_t& count, int part_id) {
  if (count == 0) return RCOK;

  auto idx = reinterpret_cast<concurrent_mbtree*>(btree_idx[part_id]);

  // mbtree's range is left-open.
  assert(min_key != 0);
  min_key--;
  u64_varkey mbtree_key_min(min_key);

  u64_varkey mbtree_key_max(max_key);

  uint64_t i = 0;
  auto cb = IndexMBTree_cb(txn, rows, count, i);

  idx->rsearch_range_call(mbtree_key_max, &mbtree_key_min, cb);
  if (cb.need_to_abort()) return Abort;

  count = i;
  // printf("%" PRIu64 "\n", i);

  return RCOK;
}

RC IndexMBTree::index_remove(txn_man* txn, idx_key_t key, row_t*, int part_id) {
  auto idx = reinterpret_cast<concurrent_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

  if (!idx->remove(mbtree_key, NULL)) return ERROR;

  return RCOK;
}

RC IndexMBTree::validate(txn_man* txn) {
#if TPCC_VALIDATE_NODE

  for (auto it : txn->node_map) {
    auto n = (concurrent_mbtree::node_opaque_t*)it.first;
    if (concurrent_mbtree::ExtractVersionNumber(n) != it.second) {
      // printf("node wts validation failure!\n");
      return Abort;
    }
  }

// printf("node validation succeeded\n");

#endif  // TPCC_VALIDATE_NODE

  return RCOK;
}
