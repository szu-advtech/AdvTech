#pragma once
#ifndef MICA_TRANSACTION_BTREE_INDEX_IMPL_NON_TRANSACTIONAL_OPERATOR_H_
#define MICA_TRANSACTION_BTREE_INDEX_IMPL_NON_TRANSACTIONAL_OPERATOR_H_

namespace mica {
namespace transaction {

// non_transactioanl_insert -------------------------------------------------------

template <class StaticConfig, bool HasValue, class Key, class Compare>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_insert(
    Context *ctx, const Key& key, uint64_t value) {

  uint64_t rah_head = 0;
  auto head = as_internal(get_node(rah_head)); // tx.peek_row
  if (!head) return kHaveToAbort;

  uint64_t rah_root = head->child_row_id(0);
  auto root_b =
      get_node_with_fixup<false, false>(rah_root, rah_root, key);
  if (!root_b) return kHaveToAbort;

  Key up_key_from_child{};
  uint64_t up_row_id_from_child = kNullRowID;

  auto ret = insert_recursive(ctx, rah_root, root_b, key, value,
                              &up_key_from_child, &up_row_id_from_child);

  if (ret == kHaveToAbort) return kHaveToAbort;

  if (up_row_id_from_child != kNullRowID) {
    if ((kVerbose & VerboseFlag::kInsert))
      printf("BTreeIndex::insert(): key=%" PRIu64
             " up_key from child; up_key = %" PRIu64 " up_row_id = %" PRIu64
             "\n",
             key_info(key), key_info(up_key_from_child), up_row_id_from_child);

    // Replace the root.
    uint64_t rah_head = 0;
    if (!get_node(rah_head)) return kHaveToAbort;
    auto head = as_internal(get_writable_node(rah_head));
    if (!head) return kHaveToAbort;

    uint64_t rah_root;
    auto root = make_internal_node(ctx, rah_root); // rah_root.new_row
    if (!root) return kHaveToAbort;

    // Left child is the original root.
    // Right child is the node created by splitting the original root.
    root->count = 1;
    if (kUseIndirection) root->indir[0] = 0;
    root->key(0) = up_key_from_child;
    // Do not use a fixed-up node (previous rah_root and root_b); the root's row ID comes directly from the head node.
    root->child_row_id(0) = head->child_row_id(0);
    root->child_row_id(1) = up_row_id_from_child;
    root->next = kNullRowID;
    root->min_key = Key{};
    root->max_key = Key{};

    head->count = 0;
    head->child_row_id(0) = rah_root;

    if ((kVerbose & VerboseFlag::kInsert))
      printf("BTreeIndex::insert(): new root node_row_id=%" PRIu64 "\n",
             head->child_row_id(0));
  }
  return ret;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
typename BTreeIndex<StaticConfig, HasValue, Key, Compare>::Node*
BTreeIndex<StaticConfig, HasValue, Key, Compare>::get_node(
    uint64_t row_id) {
  auto rv = idx_tbl_->head(0, row_id)->older_rv;
  return reinterpret_cast<Node*>(rv->data);
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <bool RightOpen, bool RightExclusive>
typename BTreeIndex<StaticConfig, HasValue, Key, Compare>::Node*
BTreeIndex<StaticConfig, HasValue, Key, Compare>::get_node_with_fixup(
    uint64_t &rah, uint64_t row_id, const Key& key) {
  auto node_b = get_node (row_id);
  if (is_internal(node_b)) {
    if (!fixup_internal<RightOpen, RightExclusive>(rah, node_b, key))
      return nullptr;
  } else {
    if (!fixup_leaf<RightOpen, RightExclusive>(rah, node_b, key))
      return nullptr;
  }
  return node_b;
}

// rah可能发生变化改变行，因此调用此函数前需要保留 rah原值，和调用后对比，以正确
template <class StaticConfig, bool HasValue, class Key, class Compare>
template <bool RightOpen, bool RightExclusive>
bool BTreeIndex<StaticConfig, HasValue, Key, Compare>::fixup_internal(
    uint64_t &rah, Node*& node_b, const Key& key) {
  // not use, for consistent interface
  (void)rah;

  auto node = as_internal(node_b);

  if (RightOpen) (void)key;

  // Ensure key < node->max_key.
  uint64_t fixup_len = 0;
  while (node->next != kNullRowID) {
    if (RightOpen)
      ;
    else if (RightExclusive) {
      if (comp_le(key, node->max_key)) break;
    } else {
      if (comp_lt(key, node->max_key)) break;
    }

    node_b = get_node(node->next);

    if (!node_b) return false;

    node = as_internal(node_b);

    if ((kVerbose & VerboseFlag::kFixup)) fixup_len++;
  }

  if (!comp_le(node->min_key, key)) return false;

  if ((kVerbose & VerboseFlag::kFixup))
    if (fixup_len > 100)
      printf("BTreeIndex::fixup_internal(): key=%" PRIu64 " fixup_len=%" PRIu64
             "\n",
             key_info(key), fixup_len);

  return true;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <bool RightOpen, bool RightExclusive>
bool BTreeIndex<StaticConfig, HasValue, Key, Compare>::fixup_leaf(
    uint64_t& rah, Node*& node_b, const Key& key) {
  auto node = as_leaf(node_b);

  // Ensure key < node->max_key.
  uint64_t fixup_len = 0;

  if (!RightOpen) {
    while (node->prev != kNullRowID) {
      if (RightExclusive) {
        if (comp_lt(node->min_key, key)) break;
      } else {
        if (comp_le(node->min_key, key)) break;
      }

      rah = node->prev;
      node_b = get_node(node->prev);

      if (!node_b) return false;

      node = as_leaf(node_b);

      if ((kVerbose & VerboseFlag::kFixup)) fixup_len++;
    }
  }

  while (node->next != kNullRowID) {
    if (RightOpen)
      ;
    else if (RightExclusive) {
      if (comp_le(key, node->max_key)) break;
    } else {
      if (comp_lt(key, node->max_key)) break;
    }

    rah = node->next;
    node_b = get_node(node->next);
    if (!node_b) return false;

    node = as_leaf(node_b);

    if ((kVerbose & VerboseFlag::kFixup)) fixup_len++;
  }

  if (!comp_le(node->min_key, key)) return false;

  if ((kVerbose & VerboseFlag::kFixup))
    if (fixup_len > 100)
      printf("BTreeIndex::fixup_leaf(): key=%" PRIu64 " fixup_len=%" PRIu64
             "\n",
             key_info(key), fixup_len);

  return true;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::insert_recursive(
    Context *ctx, uint64_t &rah, Node* node_b, const Key& key,
    uint64_t value, Key* up_key, uint64_t* up_row_id) {

  uint64_t ret;

  if (is_internal(node_b)) {

    auto node = as_internal(node_b);
    size_t j = static_cast<size_t>(search_leftmost<true>(node, key));
    auto child_row_id = node->child_row_id(j);

    uint64_t rah_child = child_row_id;  //  重要
    auto child_b =
        get_node_with_fixup<false, false>(rah_child, child_row_id, key);
    if (!child_b) return kHaveToAbort;
    child_row_id = rah_child;

    Key up_key_from_child{};
    uint64_t up_row_id_from_child = kNullRowID;
    ret = insert_recursive(ctx, rah_child, child_b, key, value,
                           &up_key_from_child, &up_row_id_from_child);

    if (ret == kHaveToAbort) return kHaveToAbort;

    if (up_row_id_from_child != kNullRowID) {
      if ((kVerbose & VerboseFlag::kInsert))
        printf("BTreeIndex::insert_recursive(): key=%" PRIu64
               " up_key from child; "
               "up_key=%" PRIu64 " up_row_id=%" PRIu64 "\n",
               key_info(key), key_info(up_key_from_child),
               up_row_id_from_child);

      if (!fixup_internal<false, false>(rah, node_b, up_key_from_child))
        return kHaveToAbort;
      node = as_internal(node_b);

      if (!insert_child(ctx, rah, node, up_key_from_child, up_row_id_from_child,
                        up_key, up_row_id))
        return kHaveToAbort;
    }
  } else {
    auto node = as_leaf(node_b);

    ret = insert_item(ctx, rah, node, key, value, up_key, up_row_id);
    // 将键值对插入到叶子节点
  }
  return ret;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::insert_item(
    Context *ctx, uint64_t &rah, LeafNode* node_r,
    const Key& key, uint64_t value, Key* up_key, uint64_t* up_row_id) {

  // Validate the leaf node because we may read it but not write to it.
  size_t j = static_cast<size_t>(search_leftmost<false>(node_r, key));

  if (j != node_r->count && comp_ge(key, node_r->key(j))) {
    // i.e., node_r->key(j) == key
    assert(comp_eq(node_r->key(j), key));
    return 0;
  }

  // Obtain a writable version of node.
  auto node = as_leaf(get_writable_node(rah));
  if (!node) return kHaveToAbort;
  node_r = node;

  if ((kVerbose & VerboseFlag::kInsert))
    printf("BTreeIndex::insert_item(): key=%" PRIu64 " node->count=%" PRIu8
           "\n",
           key_info(key), node->count);

  if (node->count == LeafNode::kMaxCount) {
    // Split the full leaf node.
    uint64_t rah_right;
    auto right = make_leaf_node(ctx, rah_right);
    if (!right) return kHaveToAbort;

    size_t new_left_count = node->count / 2;

    if ((kVerbose & VerboseFlag::kInsert))
      printf("BTreeIndex::insert_item(): key=%" PRIu64 " splitting\n",
             key_info(key));

    LeafNodeBuffer buf;
    copy(&buf, node);

    // Insert the new key at j.
    if (kUseIndirection) {
      ::mica::util::memmove(
          buf.indir + j + 1, buf.indir + j,
          sizeof(uint8_t) * (static_cast<size_t>(buf.count) - j));
      buf.indir[j] = buf.count;
    } else {
      // for (size_t i = static_cast<size_t>(buf.count); i > j; i--)
      //   buf.keys[i] = buf.keys[i - 1];
      ::mica::util::memmove(buf.keys + j + 1, buf.keys + j,
                            sizeof(Key) * (static_cast<size_t>(buf.count) - j));
      if (HasValue)
        ::mica::util::memmove(
            buf.values + j + 1, buf.values + j,
            sizeof(uint64_t) * (static_cast<size_t>(buf.count) - j));
    }
    buf.key(j) = key;
    if (HasValue) buf.value(j) = value;
    buf.count++;

    // Adjust the left node size if the new child will be on the left node.
    if (j < new_left_count) new_left_count++;

    scatter(node, right, &buf, new_left_count);

    // Fix the doubly-linked list.
    right->next = node->next;
    node->next = rah_right;

    right->prev = rah;
    if (right->next != kNullRowID) {
      uint64_t rah_right_right = right->next;
      if (!get_node(right->next)) return kHaveToAbort;
      auto right_right = as_leaf(get_writable_node(rah_right_right));
      if (!right_right) return kHaveToAbort;
      if (right_right->prev != rah) return kHaveToAbort;
      right_right->prev = node->next;  // == rah_right.row_id();
    }

    // The first key of the (new) right node needs to be inserted in the
    // parent node.
    *up_key = right->key(0);
    *up_row_id = rah_right;
  } else {
    // Insert the new key at j.
    if (kUseIndirection) {
      ::mica::util::memmove(
          node->indir + j + 1, node->indir + j,
          sizeof(uint8_t) * (static_cast<size_t>(node->count) - j));
      node->indir[j] = node->count;
    } else {
      // for (size_t i = static_cast<size_t>(node->count); i > j; i--)
      //   node->keys[i] = node->keys[i - 1];
      ::mica::util::memmove(
          node->keys + j + 1, node->keys + j,
          sizeof(Key) * (static_cast<size_t>(node->count) - j));
      if (HasValue)
        ::mica::util::memmove(
            node->values + j + 1, node->values + j,
            sizeof(uint64_t) * (static_cast<size_t>(node->count) - j));
    }
    node->key(j) = key;
    if (HasValue) node->value(j) = value;
    node->count++;

  }
  return 1;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
bool BTreeIndex<StaticConfig, HasValue, Key, Compare>::insert_child(
    Context *ctx, uint64_t &rah, InternalNode* node_r,
    const Key& key, uint64_t child_row_id, Key* up_key, uint64_t* up_row_id) {

  // Find the insert position for the split key.
  size_t j = static_cast<size_t>(search_leftmost<true>(node_r, key));

  // Obtain a writable version of node.
  auto node = as_internal(get_writable_node(rah));
  if (!node) return false;
  node_r = node;
  (void)node_r;

  if (node->count == InternalNode::kMaxCount) {

    uint64_t rah_right;
    auto right = make_internal_node(ctx, rah_right);
    if (!right) return false;

    size_t new_left_count = (static_cast<size_t>(node->count) - 1) / 2;

    if ((kVerbose & VerboseFlag::kInsert))
      printf("BTreeIndex::insert_child(): key=%" PRIu64 " splitting\n",
             key_info(key));

    InternalNodeBuffer buf;
    copy(&buf, node);

    // Insert the new key at j.
    if (kUseIndirection) {
      ::mica::util::memmove(
          buf.indir + j + 1, buf.indir + j,
          sizeof(uint8_t) * (static_cast<size_t>(buf.count) - j));
      buf.indir[j] = buf.count;
    } else {
      // for (size_t i = static_cast<size_t>(buf.count); i > j; i--)
      //   buf.keys[i] = buf.keys[i - 1];
      ::mica::util::memmove(buf.keys + j + 1, buf.keys + j,
                            sizeof(Key) * (static_cast<size_t>(buf.count) - j));
      ::mica::util::memmove(
          buf.child_row_ids + j + 2, buf.child_row_ids + j + 1,
          sizeof(uint64_t) * (static_cast<size_t>(buf.count) - j));
    }
    buf.key(j) = key;
    buf.child_row_id(j + 1) = child_row_id;
    buf.count++;

    // Adjust the left node size if the new child will be on the left node.
    if (j < new_left_count) new_left_count++;

    scatter(node, right, &buf, new_left_count);

    right->next = node->next;
    node->next = rah_right;

    // The last key of the left node needs to be inserted in the
    // parent node so that the right node can be found.
    // This key will not be used in this node in the future.
    *up_key = node->max_key;
    *up_row_id = rah_right;
  } else {
    // Insert the new key at j.
    if (kUseIndirection) {
      ::mica::util::memmove(
          node->indir + j + 1, node->indir + j,
          sizeof(uint8_t) * (static_cast<size_t>(node->count) - j));
      node->indir[j] = node->count;
    } else {
      // for (size_t i = static_cast<size_t>(node->count); i > j; i--)
      //   node->keys[i] = node->keys[i - 1];
      ::mica::util::memmove(
          node->keys + j + 1, node->keys + j,
          sizeof(Key) * (static_cast<size_t>(node->count) - j));
      ::mica::util::memmove(
          node->child_row_ids + j + 2, node->child_row_ids + j + 1,
          sizeof(uint64_t) * (static_cast<size_t>(node->count) - j));
    }
    node->key(j) = key;
    node->child_row_id(j + 1) = child_row_id;
    node->count++;

  }
  return true;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
typename BTreeIndex<StaticConfig, HasValue, Key, Compare>::Node*
BTreeIndex<StaticConfig, HasValue, Key, Compare>::get_writable_node(
    uint64_t &rah) {
  return get_node (rah);
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
typename BTreeIndex<StaticConfig, HasValue, Key, Compare>::InternalNode*
BTreeIndex<StaticConfig, HasValue, Key, Compare>::make_internal_node(
    Context *ctx, uint64_t& rah) {

  rah = ctx->allocate_row (idx_tbl_);

  if (rah == static_cast<uint64_t>(-1))
    return nullptr;

  // 获得分配行之后，并没有实体存储空间，需要自行分配
  auto head = idx_tbl_->head (0, rah);
  auto rv = ctx->allocate_version_for_new_row(
                 idx_tbl_, 0, rah, head, kDataSize);
  rv->wts.t2 = 0UL;
  rv->rts.t2 = 0UL;
  rv->status = RowVersionStatus::kCommitted;
  head->older_rv = rv;

  auto node = reinterpret_cast<InternalNode*>(get_node(rah));

  node->type = NodeType::kInternal;
  return node;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
typename BTreeIndex<StaticConfig, HasValue, Key, Compare>::LeafNode*
BTreeIndex<StaticConfig, HasValue, Key, Compare>::make_leaf_node(
    Context *ctx, uint64_t& rah) {

  rah = ctx->allocate_row (idx_tbl_);

  if (rah == static_cast<uint64_t>(-1))
    return nullptr;

  // 获得分配行之后，并没有实体存储空间，需要自行分配
  auto head = idx_tbl_->head (0, rah);
  auto rv = ctx->allocate_version_for_new_row (
                 idx_tbl_, 0, rah, head, kDataSize);
  rv->wts.t2 = 0UL;
  rv->rts.t2 = 0UL;
  rv->status = RowVersionStatus::kCommitted;
  head->older_rv = rv;

  auto node = reinterpret_cast<LeafNode*>(get_node(rah));

  node->type = NodeType::kLeaf;
  return node;
}

// non_transactional_update--------------------------------------------------------

template <class StaticConfig, bool HasValue, class Key, class Compare>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_update(
    const Key& key, uint64_t new_value) {
  uint64_t look_result[16], pos = 0;
  auto func = [&look_result, &pos] (auto &k, auto &v) {
    (void)k;
    look_result[pos++] = v;
    return false;
  };
  uint64_t lookup_num = non_transactional_lookup_ptr (key, func);
  for (uint64_t i = 0;i < pos; i++) {
    *reinterpret_cast<uint64_t*>(look_result[i]) = new_value;
  }
  return lookup_num;
}

// non_transactional_lookup

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <typename Func>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_lookup(
    const Key& key, const Func& func) {
  return lookup<BTreeRangeType::kInclusive, BTreeRangeType::kInclusive, false, false>
         (key, key, func);
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <typename Func>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_lookup_ptr(
    const Key& key, const Func& func) {
  return lookup<BTreeRangeType::kInclusive, BTreeRangeType::kInclusive, false, true>
         (key, key, func);
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <BTreeRangeType LeftRangeType, BTreeRangeType RightRangeType,
          bool Reversed, bool GetPtr, typename Func>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::lookup(
    const Key& min_key, const Key& max_key,
    const Func& func) {

  if ((kVerbose & VerboseFlag::kLookup))
    printf("BTreeIndex::lookup(): min_key=%" PRIu64 " max_key=%" PRIu64 "\n",
           key_info(min_key), key_info(max_key));

  uint64_t rah = 0;
  auto head = as_internal(get_node(0));
  if (!head) return kHaveToAbort;

  return lookup_recursive<LeftRangeType, RightRangeType, Reversed, GetPtr>(
        rah, head, min_key, max_key, func);
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <BTreeRangeType LeftRangeType, BTreeRangeType RightRangeType,
          bool Reversed, bool GetPtr, typename Func>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::lookup_recursive(
    uint64_t &rah, const Node* node_b,
    const Key& min_key, const Key& max_key,
    const Func& func) {

  if (is_internal(node_b)) {
    auto node = as_internal(node_b);

    // Search for the first matching child.
    int j;
    if (!Reversed) {
      if (LeftRangeType == BTreeRangeType::kOpen)
        j = 0;
      else
        j = search_leftmost<true>(node, min_key);
    } else {
      if (RightRangeType == BTreeRangeType::kOpen)
        j = node->count;
      else if (RightRangeType == BTreeRangeType::kInclusive)
        j = search_rightmost<false>(node, max_key) + 1;
      else /* if (RightRangeType == BTreeRangeType::kExclusive) */
        j = search_rightmost<true>(node, max_key) + 1;
    }

    auto child_row_id = node->child_row_id(j);

    uint64_t rah_child = child_row_id;
    Node* child_b;
    if (!Reversed) {
      if (LeftRangeType == BTreeRangeType::kOpen)
        child_b = get_node(child_row_id);
      else
        child_b =
            get_node_with_fixup<false, false>(child_row_id, child_row_id, min_key);
    } else {
      if (RightRangeType == BTreeRangeType::kOpen)
        child_b =
            get_node_with_fixup<true, false>(child_row_id,child_row_id ,max_key);
      else if (RightRangeType == BTreeRangeType::kInclusive) {
        child_b =
            get_node_with_fixup<false, false>(child_row_id, child_row_id, max_key);
      } else /* if (RightRangeType == BTreeRangeType::kExclusive) */ {
        child_b =
            get_node_with_fixup<false, true>(child_row_id, child_row_id, max_key);
      }
    }
    if (!child_b) return kHaveToAbort;

    return lookup_recursive<LeftRangeType, RightRangeType, Reversed, GetPtr>(
        rah_child, child_b, min_key, max_key, func);
  } else {
    return return_range<LeftRangeType, RightRangeType, Reversed, GetPtr>(
        rah, node_b, min_key, max_key, func);
  }
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
template <BTreeRangeType LeftRangeType, BTreeRangeType RightRangeType,
          bool Reversed, bool GetPtr, typename Func>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::return_range(
    uint64_t &rah, const Node* node_b, const Key& min_key,
    const Key& max_key, const Func& func) {
  auto node = as_leaf(node_b);

  // Search for the first matching key.
  int j;
  if (!Reversed) {
    if (LeftRangeType == BTreeRangeType::kOpen) {
      j = 0;
      assert(node->prev == kNullRowID);
    } else if (LeftRangeType == BTreeRangeType::kInclusive) {
      j = search_leftmost<false>(node, min_key);

      assert(j - 1 < 0 || j - 1 >= node->count ||
             comp_lt(node->key(j - 1), min_key));
      assert(j < 0 || j >= node->count || comp_le(min_key, node->key(j)));
    } else /*if (LeftRangeType == BTreeRangeType::kExclusive)*/ {
      j = search_leftmost<true>(node, min_key);

      assert(j - 1 < 0 || j - 1 >= node->count ||
             comp_le(node->key(j - 1), min_key));
      assert(j < 0 || j >= node->count || comp_lt(min_key, node->key(j)));
    }
  } else {
    if (RightRangeType == BTreeRangeType::kOpen) {
      j = node->count - 1;
      assert(node->next == kNullRowID);
    } else if (RightRangeType == BTreeRangeType::kInclusive) {
      j = search_rightmost<false>(node, max_key);

      assert(j + 1 < 0 || j + 1 >= node->count ||
             comp_lt(max_key, node->key(j + 1)));
      assert(j < 0 || j >= node->count || comp_le(node->key(j + 1), max_key));
    } else /*if (RightRangeType == BTreeRangeType::kExclusive)*/ {
      j = search_rightmost<true>(node, max_key);

      assert(j + 1 < 0 || j + 1 >= node->count ||
             comp_le(max_key, node->key(j + 1)));
      assert(j < 0 || j >= node->count || comp_lt(node->key(j + 1), max_key));
    }
  }

  uint64_t found = 0;
  bool done = false;

  while (true) {
    if (!Reversed) {
      for (; j < node->count; j++) {
        bool matching;
        if (RightRangeType == BTreeRangeType::kOpen)
          matching = true;
        else if (RightRangeType == BTreeRangeType::kInclusive)
          matching = comp_le(node->key(j), max_key);
        else /*if (RightRangeType == BTreeRangeType::kExclusive)*/
          matching = comp_lt(node->key(j), max_key);

        if (matching) {
          found++;

          uint64_t value;
          if (HasValue) {
            if (GetPtr)
              value = reinterpret_cast<uint64_t>(&(node->value(j)));
            else
              value = node->value(j);
          }
          else
            value = 0;

          if (!func(node->key(j), value)) {
            done = true;
            break;
          }
        } else {
          done = true;
          break;
        }
      }
      if (done) break;
      if (node->next == kNullRowID) break;
    } else {

      for (; j >= 0; j--) {
        bool matching;
        if (LeftRangeType == BTreeRangeType::kOpen)
          matching = true;
        else if (LeftRangeType == BTreeRangeType::kInclusive)
          matching = comp_le(min_key, node->key(j));
        else /*if (LeftRangeType == BTreeRangeType::kExclusive)*/
          matching = comp_lt(min_key, node->key(j));

        if (matching) {
          found++;

          uint64_t value;
          if (HasValue) {
            if (GetPtr)
              value = reinterpret_cast<uint64_t>(&(node->value(j)));
            else
              value = node->value(j);
          }
          else
            value = 0;

          if (!func(node->key(j), value)) {
            done = true;
            break;
          }
        } else {
          done = true;
          break;
        }
      }
      if (done) break;
      if (node->prev == kNullRowID) break;
    }

    // TODO: Use node->min_key and node->max_key to terminate early without touching the sibling node.

    rah = node->next;
    if (!Reversed)
      node = as_leaf(get_node(node->next));
    else
      node = as_leaf(get_node(node->prev));
    if (!node) return kHaveToAbort;

    if (!Reversed)
      j = 0;
    else
      j = node->count - 1;
  }

  return found;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
uint64_t BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_remove(
    Context *ctx, const Key& key, uint64_t value) {
  return 0;
}

template <class StaticConfig, bool HasValue, class Key, class Compare>
void BTreeIndex<StaticConfig, HasValue, Key, Compare>::non_transactional_check(
    void) {
}

}
}

#endif

