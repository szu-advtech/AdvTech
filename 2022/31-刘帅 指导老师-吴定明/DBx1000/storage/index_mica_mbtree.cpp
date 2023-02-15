#define CONFIG_H "silo/config/config-perf.h"

#define NDB_MASSTREE 1
#include "silo/masstree/config.h"
#include "silo/masstree_btree.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "global.h"
#include "index_mica_mbtree.h"
#include "mem_alloc.h"
#include "table.h"
#include "row.h"
#include "helper.h"
#include "catalog.h"
#include "txn.h"

#if INDEX_STRUCT == IDX_MICA

struct mica_mbtree_value_type {
  row_t* row;
  int refcount;
};

struct mica_mbtree_params : public Masstree::nodeparams<> {
  typedef mica_mbtree_value_type value_type;
  typedef Masstree::value_print<value_type> value_print_type;
  typedef simple_threadinfo threadinfo_type;
  enum { RcuRespCaller = true };
};

typedef mbtree<mica_mbtree_params> concurrent_mica_mbtree;

class IndexMICAMBTree_cb
    : public concurrent_mica_mbtree::low_level_search_range_callback {
 public:
  IndexMICAMBTree_cb(txn_man* txn, row_t** rows, uint64_t count, uint64_t& i)
      : txn_(txn), rows_(rows), count_(count), i_(i), abort_(false) {}

  void on_resp_node(const concurrent_mica_mbtree::node_opaque_t* n,
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

  bool invoke(const concurrent_mica_mbtree::string_type& k,
              concurrent_mica_mbtree::value_type v,
              const concurrent_mica_mbtree::node_opaque_t* n,
              uint64_t version) override {
    (void)k;
    (void)n;
    (void)version;
    rows_[i_++] = v.row;
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

RC IndexMICAMBTree::init(uint64_t part_cnt, table_t* table) {
  return init(part_cnt, table, 0);
}

RC IndexMICAMBTree::init(uint64_t part_cnt, table_t* table,
                         uint64_t bucket_cnt) {
  (void)bucket_cnt;

  this->table = table;

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    mem_allocator.register_thread(part_id % g_thread_cnt);

    auto t = (concurrent_mica_mbtree*)mem_allocator.alloc(
        sizeof(concurrent_mica_mbtree), part_id);
    new (t) concurrent_mica_mbtree;

    btree_idx.push_back(t);
  }

#if TPCC_VALIDATE_GAP
  validate_gap = false;
#endif

  return RCOK;
}

RC IndexMICAMBTree::index_insert(txn_man* txn, MICATransaction* tx,
                                 idx_key_t key, row_t* row, int part_id) {
  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

  mica_mbtree_value_type v{row, 1};
#if !TPCC_VALIDATE_NODE
  if (!idx->insert_refcount(mbtree_key, v)) return ERROR;
#else
  concurrent_mica_mbtree::insert_info_t insert_info;
  if (!idx->insert_refcount(mbtree_key, v, &insert_info)) return ERROR;

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

RC IndexMICAMBTree::index_read(txn_man* txn, idx_key_t key, row_t** row,
                               int part_id) {
  // auto tx = txn->mica_tx;
  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

  mica_mbtree_value_type v;

  concurrent_mica_mbtree::versioned_node_t search_info;
  if (!idx->search(mbtree_key, v, &search_info)) {
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

  *row = v.row;

#if TPCC_VALIDATE_GAP
// TODO: Implement.
#endif

  return RCOK;
}

RC IndexMICAMBTree::index_read_multiple(txn_man* txn, idx_key_t key,
                                        row_t** rows, uint64_t& count,
                                        int part_id) {
  // Duplicate keys are currently not supported in IndexMBTree.
  assert(false);
  (void)txn;
  (void)key;
  (void)rows;
  (void)count;
  (void)part_id;
  return ERROR;
}

RC IndexMICAMBTree::index_read_range(txn_man* txn, idx_key_t min_key,
                                     idx_key_t max_key, row_t** rows,
                                     uint64_t& count, int part_id) {
  if (count == 0) return RCOK;
  // auto tx = txn->mica_tx;

  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key_min(min_key);

  // mbtree's range is right-open.
  max_key++;
  assert(max_key != 0);
  u64_varkey mbtree_key_max(max_key);

  // MICARowAccessHandlePeekOnly rah(tx);
  // MICARowAccessHandle rah(tx);
  // auto mica_tbl = table->mica_tbl[part_id];

  uint64_t i = 0;
  auto cb = IndexMICAMBTree_cb(txn, rows, count, i);

  // auto f = [&i, rows, count, [>&rah,<] mica_tbl](auto& k, auto& v) {
  //   // if (!rah.peek_row(mica_tbl, 0, (uint64_t)v.row, false, false, false)) return true;
  //   // if (!rah.peek_row(mica_tbl, 0, (uint64_t)v.row, true, false, false)) return true;
  //   // rah.reset();
  //
  //   rows[i++] = v.row;
  //   return i < count;
  // };

  idx->search_range_call(mbtree_key_min, &mbtree_key_max, cb);
  if (cb.need_to_abort()) return Abort;

  // Validating read index entries' value is not necessary if the indexed value
  // does not change.  Whether rows are visible is checked in the application
  // code.

  count = i;

#if TPCC_VALIDATE_GAP
  if (!tx->is_peek_only() && validate_gap) {
    MICARowAccessHandle rah(tx);
    if (count == 0) {
      if (!rah.peek_row(mica_tbl, 1, pos_infty, false, true, false) ||
          !rah.read_row())
        return Abort;
    } else {
      for (uint64_t i = 0; i < count; i++) {
        if (!rah.peek_row(mica_tbl, 1, (uint64_t)rows[i], false, true, false) ||
            !rah.read_row())
          return Abort;
        rah.reset();
      }
    }
  }
#endif

  return RCOK;
}

RC IndexMICAMBTree::index_read_range_rev(txn_man* txn, idx_key_t min_key,
                                         idx_key_t max_key, row_t** rows,
                                         uint64_t& count, int part_id) {
  if (count == 0) return RCOK;
  // auto tx = txn->mica_tx;

  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);

  // mbtree's range is left-open.
  assert(min_key != 0);
  min_key--;
  u64_varkey mbtree_key_min(min_key);

  u64_varkey mbtree_key_max(max_key);

  // MICARowAccessHandlePeekOnly rah(tx);
  // MICARowAccessHandle rah(tx);
  // auto mica_tbl = table->mica_tbl[part_id];

  uint64_t i = 0;
  auto cb = IndexMICAMBTree_cb(txn, rows, count, i);

  // auto f = [&i, rows, count, [>&rah,<] mica_tbl](auto& k, auto& v) {
  //   // if (!rah.peek_row(mica_tbl, 0, (uint64_t)v.row, false, false, false)) return true;
  //   // if (!rah.peek_row(mica_tbl, 0, (uint64_t)v.row, true, false, false)) return true;
  //   // rah.reset();
  //
  //   rows[i++] = v.row;
  //   return i < count;
  // };

  idx->rsearch_range_call(mbtree_key_max, &mbtree_key_min, cb);
  if (cb.need_to_abort()) return Abort;

  // Validating read index entries' value is not necessary if the indexed value
  // does not change.  Whether rows are visible is checked in the application
  // code.

  count = i;
  // printf("%" PRIu64 "\n", i);

#if TPCC_VALIDATE_GAP
  if (!tx->is_peek_only() && validate_gap) {
    MICARowAccessHandle rah(tx);
    if (count == 0) {
      if (!rah.peek_row(mica_tbl, 1, neg_infty, false, true, false) ||
          !rah.read_row())
        return Abort;
    } else {
      for (uint64_t i = 0; i < count; i++) {
        if (!rah.peek_row(mica_tbl, 1, (uint64_t)rows[i], false, true, false) ||
            !rah.read_row())
          return Abort;
        rah.reset();
      }
    }
  }
#endif

  return RCOK;
}

RC IndexMICAMBTree::index_remove(txn_man* txn, MICATransaction* tx,
                                 idx_key_t key, row_t*, int part_id) {
  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);

  u64_varkey mbtree_key(key);

  if (!idx->remove_refcount(mbtree_key, NULL)) return ERROR;

  return RCOK;
}

#if TPCC_VALIDATE_GAP
RC IndexMICAMBTree::list_init(MICADB* mica_db, uint64_t gap_off) {
  // printf("gap_off=%" PRIu64 "\n", gap_off);

  validate_gap = true;
  this->gap_off = gap_off;

  auto thread_id = ::mica::util::lcore.lcore_id();

  MICATransaction tx(mica_db->context(thread_id));

  for (uint64_t part_id = 0; part_id < btree_idx.size(); part_id++) {
    auto mica_tbl = table->mica_tbl[part_id];
    auto tuple_size = table->get_schema()->get_tuple_size();

    // Create -infty, +infty.
    MICARowAccessHandle rah0(&tx);
    MICARowAccessHandle rah1(&tx);

    if (!tx.begin()) assert(false);

    if (!rah0.new_row(mica_tbl, 0, MICATransaction::kNewRowID, false,
                      tuple_size))
      assert(false);
    if (!rah1.new_row(mica_tbl, 0, MICATransaction::kNewRowID, false,
                      tuple_size))
      assert(false);

    if (part_id == 0) {
      neg_infty = rah0.row_id();
      pos_infty = rah1.row_id();
    } else {
      assert(neg_infty == rah0.row_id());
      assert(pos_infty == rah1.row_id());
    }

    rah0.reset();
    rah1.reset();
    if (!rah0.new_row(mica_tbl, 1, neg_infty, false, 16)) assert(false);
    if (!rah1.new_row(mica_tbl, 1, pos_infty, false, 16)) assert(false);

    *(uint64_t*)(rah0.data() + gap_off + 0) = (uint64_t)-1;

    *(uint64_t*)(rah1.data() + gap_off + 8) = (uint64_t)-1;

    if (!tx.commit()) assert(false);
  }

  return RCOK;
}

RC IndexMICAMBTree::list_make(MICADB* mica_db) {
  assert(validate_gap);

  auto thread_id = ::mica::util::lcore.lcore_id();

  MICATransaction tx(mica_db->context(thread_id));

  for (uint64_t part_id = 0; part_id < btree_idx.size(); part_id++) {
    auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);
    auto mica_tbl = table->mica_tbl[part_id];

    u64_varkey mbtree_key_min(0);

    uint64_t prev_row = neg_infty;

    // Create the main chain.
    auto f = [&tx, &mica_tbl, &prev_row, this](auto& k, auto& v) {
      MICARowAccessHandle rah_left(&tx);
      MICARowAccessHandle rah(&tx);
      uint64_t row_id = (uint64_t)v.row;

      if (!tx.begin()) assert(false);

      if (!rah_left.peek_row(mica_tbl, 1, prev_row, false, true, true) ||
          !rah_left.read_row() || !rah_left.write_row())
        assert(false);
      if (!rah.new_row(mica_tbl, 1, row_id, false, 16)) assert(false);

      *(uint64_t*)(rah_left.data() + gap_off + 8) = row_id;
      *(uint64_t*)(rah.data() + gap_off + 0) = prev_row;
      prev_row = row_id;

      if (!tx.commit()) assert(false);
      return true;
    };

    idx->search_range(mbtree_key_min, NULL, f);

    {
      MICARowAccessHandle rah_left(&tx);
      MICARowAccessHandle rah(&tx);

      if (!tx.begin()) assert(false);

      if (!rah_left.peek_row(mica_tbl, 1, prev_row, false, true, true) ||
          !rah_left.read_row() || !rah_left.write_row())
        assert(false);
      if (!rah.peek_row(mica_tbl, 1, pos_infty, false, true, true) ||
          !rah.read_row() || !rah.write_row())
        assert(false);

      *(uint64_t*)(rah_left.data() + gap_off + 8) = pos_infty;
      *(uint64_t*)(rah.data() + gap_off + 0) = prev_row;
      *(uint64_t*)(rah.data() + gap_off + 8) = (uint64_t)-1;

      if (!tx.commit()) assert(false);
    }
  }

  return RCOK;
}

RC IndexMICAMBTree::validate(txn_man* txn) {
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

RC IndexMICAMBTree::list_insert(MICATransaction* tx, idx_key_t key, row_t* row,
                                int part_id) {
  if (!validate_gap) return RCOK;

  auto idx = reinterpret_cast<concurrent_mica_mbtree*>(btree_idx[part_id]);
  auto mica_tbl = table->mica_tbl[part_id];
  uint64_t row_id = row->get_row_id();

  // Make a node.
  MICARowAccessHandle rah(tx);
  if (!rah.new_row(mica_tbl, 1, row_id, true, 16)) assert(false);

  // Find the right node.
  // TODO: Do not use the index because the same transaction may insert multiple
  // entries to the index (e.g., new_order_createOrderLine).
  uint64_t right = pos_infty;
  {
    u64_varkey mbtree_key_min(key);
    auto f = [&right](auto& k, auto& v) {
      right = (uint64_t)v.row;
      return false;
    };
    idx->search_range(mbtree_key_min, NULL, f);

    // if (right == pos_infty) printf("warning: the right node is pos_infty\n");
    // printf("right=%" PRIu64 "\n", right);
  }

  // Read the right node and find the left node.
  MICARowAccessHandle rah_right(tx);
  uint64_t left;
  {
    if (!rah_right.peek_row(mica_tbl, 1, right, true, true, true) ||
        !rah_right.read_row() || !rah_right.write_row()) {
      // printf("i right\n");
      return Abort;
    }
    left = *(uint64_t*)(rah_right.data() + gap_off + 0);
    // printf("left=%" PRIu64 "\n", left);
  }

  // Read the left node.
  MICARowAccessHandle rah_left(tx);
  if (!rah_left.peek_row(mica_tbl, 1, left, true, true, true) ||
      !rah_left.read_row() || !rah_left.write_row()) {
    // printf("i left\n");
    return Abort;
  }

  // Insert the new node.
  *(uint64_t*)(rah_left.data() + gap_off + 8) = row_id;
  *(uint64_t*)(rah.data() + gap_off + 0) = left;
  *(uint64_t*)(rah.data() + gap_off + 8) = right;
  *(uint64_t*)(rah_right.data() + gap_off + 0) = row_id;
  return RCOK;
}

RC IndexMICAMBTree::list_remove(MICATransaction* tx, idx_key_t key, row_t* row,
                                int part_id) {
  if (!validate_gap) return RCOK;

  auto mica_tbl = table->mica_tbl[part_id];
  uint64_t row_id = row->get_row_id();

  MICARowAccessHandle rah(tx);
  MICARowAccessHandle rah_left(tx);
  MICARowAccessHandle rah_right(tx);

  // Find left and right nodes.
  if (!rah.peek_row(mica_tbl, 1, row_id, true, true, true) || !rah.read_row()) {
    // printf("r node\n");
    return Abort;
  }
  auto left = *(uint64_t*)(rah.cdata() + gap_off + 0);
  auto right = *(uint64_t*)(rah.cdata() + gap_off + 8);

  // Read the left node.
  if (!rah_left.peek_row(mica_tbl, 1, left, true, true, true) ||
      !rah_left.read_row() || !rah_left.write_row()) {
    // printf("r left\n");
    return Abort;
  }
  // Read the right node.
  if (!rah_right.peek_row(mica_tbl, 1, right, true, true, true) ||
      !rah_right.read_row() || !rah_right.write_row()) {
    // printf("r right\n");
    return Abort;
  }

  // Remove the node from the chain.
  *(uint64_t*)(rah_left.data() + gap_off + 8) = right;
  *(uint64_t*)(rah_right.data() + gap_off + 0) = left;

  // Delete the node
  if (!rah.write_row(0) || !rah.delete_row()) {
    // printf("r node\n");
    return Abort;
  }

  return RCOK;
}
#endif

#endif
