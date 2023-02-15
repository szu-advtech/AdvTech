#pragma once
#ifndef MICA_TRANSACTION_HASH_INDEX_IMPL_NON_TRANSACTIONAL_OPERATOR_H_
#define MICA_TRANSACTION_HASH_INDEX_IMPL_NON_TRANSACTIONAL_OPERATOR_H_

namespace mica {
namespace transaction {

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
template <typename Func>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::non_transactional_lookup(
    const Key& key, const Func& func) {

  uint64_t found = 0;

  auto bkt_id = get_bucket_id(key);

  while (true) {

    auto rv   = idx_tbl_->head(0, bkt_id)->older_rv;
    auto bkt = reinterpret_cast<Bucket*>(rv->data);

    for (size_t j = 0; j < Bucket::kBucketSize; j++) {
      // printf("HashIndex::lookup() key=%" PRIu64 " bucket_key=%" PRIu64
      //        " value=%" PRIu64 "\n",
      //        key, bkt->keys[i], bkt->values[i]);
      if (!key_equal_(bkt->keys[j], key)) continue;

      if (bkt->values[j] == kNullRowID) continue;

      auto value = bkt->values[j];

      found++;

      if (!func(bkt->keys[j], value)) return found;

      if (UniqueKey) {
        // There will be no matching key.
        return found;
      }

      // To omit checking j if the bucket has only one item.
      if (Bucket::kBucketSize == 1) break;
    }

    bkt_id = bkt->next;
    if (bkt_id == kNullRowID) {
      return found;
    }
  }
}

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
template <typename Func>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::non_transactional_lookup_ptr (
    const Key& key, const Func& func) {

  uint64_t found = 0;
  auto bkt_id = get_bucket_id(key);

  while (true) {

    auto rv   = idx_tbl_->head(0, bkt_id)->older_rv;
    auto bkt = reinterpret_cast<Bucket*>(rv->data);

    for (size_t j = 0; j < Bucket::kBucketSize; j++) {
      // printf("HashIndex::lookup() key=%" PRIu64 " bucket_key=%" PRIu64
      //        " value=%" PRIu64 "\n",
      //        key, bkt->keys[i], bkt->values[i]);
      if (!key_equal_(bkt->keys[j], key)) continue;

      if (bkt->values[j] == kNullRowID) continue;

      auto value = reinterpret_cast<uint64_t>(&(bkt->values[j]));

      found++;

      if (!func(bkt->keys[j], value)) return found;

      if (UniqueKey) {
        // There will be no matching key.
        return found;
      }

      // To omit checking j if the bucket has only one item.
      if (Bucket::kBucketSize == 1) break;
    }

    bkt_id = bkt->next;
    if (bkt_id == kNullRowID) {
      return found;
    }
  }
}

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::non_transactional_insert (
    Context *ctx, const Key& key, uint64_t value) {

  auto bkt_id = get_bucket_id(key);
  auto rv   = idx_tbl_->head(0, bkt_id)->older_rv;
  auto cbkt = reinterpret_cast<Bucket*>(rv->data);

  while (true) {
    if (UniqueKey) {
      for (uint64_t j=0; j< Bucket::kBucketSize; j++) {
        if (cbkt->values[j] != kNullRowID && key_equal_(cbkt->keys[j], key)) {
          // 有重复的数据，不再插入数据
          return 0;
        }
      }
    }
    
    if (cbkt->next == kNullRowID) {
      break;
    }

    bkt_id = cbkt->next;

    rv   = idx_tbl_->head(0, bkt_id)->older_rv;
    cbkt = reinterpret_cast<Bucket*>(rv->data);
  }

  rv   = idx_tbl_->head(0, bkt_id)->older_rv;
  auto bkt = reinterpret_cast<Bucket*>(rv->data);
 
  uint64_t j;
  for (j = 0; j < Bucket::kBucketSize; j++) {
    if (bkt->values[j] == kNullRowID) {
      break;
    }
  }

  if (j == Bucket::kBucketSize) {
    auto new_bkt_id = ctx->allocate_row (idx_tbl_);

    // 获得分配行之后，并没有实体存储空间，需要自行分配
    auto head = idx_tbl_->head (0, new_bkt_id);
    auto rv = ctx->allocate_version_for_new_row(
                 idx_tbl_, 0, new_bkt_id, head, kDataSize);
    rv->wts.t2 = 0UL;
    rv->rts.t2 = 0UL;
    rv->status = RowVersionStatus::kCommitted;
    head->older_rv = rv;

    auto new_rv = idx_tbl_->head(0, new_bkt_id)->older_rv;
    auto new_bkt = reinterpret_cast<Bucket*>(new_rv->data);
   
    for (j = 0; j< Bucket::kBucketSize; j++) {
      new_bkt->values[j] = kNullRowID;
    }
    new_bkt->next = kNullRowID;
    j = 0;

    bkt->next = new_bkt_id;
    bkt = new_bkt;
  }

  bkt->keys[j] = key;
  bkt->values[j] = value;
  return 1;
}

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::non_transactional_update (
    const Key& key, uint64_t new_value) {

  auto bkt_id = get_bucket_id(key);
 
  auto rv   = idx_tbl_->head(0, bkt_id)->older_rv;
  auto cbkt = reinterpret_cast<Bucket*>(rv->data);

  uint64_t ret_num = 0;
  while (true) {
    if (UniqueKey) {
      for (uint64_t j=0; j< Bucket::kBucketSize; j++) {
        if (cbkt->values[j] != kNullRowID && key_equal_(cbkt->keys[j], key)) {
          // 有重复的数据，不再插入数据
          cbkt->values[j] = new_value;
          return 1;
        }
      }
    }
    else {
      for (uint64_t j=0; j< Bucket::kBucketSize; j++) {
        if (cbkt->values[j] != kNullRowID && key_equal_(cbkt->keys[j], key)) {
          // 有重复的数据，不再插入数据
          cbkt->values[j] = new_value;
          ret_num += 1;
        }
      }
    }
    
    if (cbkt->next == kNullRowID) {
      break;
    }

    bkt_id = cbkt->next;
    rv   = idx_tbl_->head(0, bkt_id)->older_rv;
    cbkt = reinterpret_cast<Bucket*>(rv->data);
  }
  return ret_num;
}

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::non_transactional_remove (
    Context *ctx, const Key& key, uint64_t value) {
  // TODO:数据恢复过程不会用到
  return 0;
}

template <class StaticConfig, bool UniqueKey, class Key, class Hash,
          class KeyEqual>
template <typename Func>
uint64_t HashIndex<StaticConfig, UniqueKey, Key, Hash, KeyEqual>::
  non_transactional_lookup_ptr_for_multiple_recovery (
  const Key& key, const Func& func, volatile uint8_t **cas_status) {

  uint64_t found = 0;
  auto bkt_id = get_bucket_id(key);

  auto hd = idx_tbl_->head(0, bkt_id);
  *cas_status = &(hd->status);

  // printf ("non_transactional_lookup_ptr_for_multiple_recovery cas_started!\n");
  while (!__sync_bool_compare_and_swap(&hd->status, 0, 1))
    ::mica::util::pause();
  // printf ("non_transactional_lookup_ptr_for_multiple_recovery cas_locked!\n");

  while (true) {
    auto rv   = idx_tbl_->head(0, bkt_id)->older_rv;
    auto bkt = reinterpret_cast<Bucket*>(rv->data);

    for (size_t j = 0; j < Bucket::kBucketSize; j++) {

      // printf("HashIndex::lookup() key=%" PRIu64 " bucket_key=%" PRIu64
      //        " value=%" PRIu64 "\n",
      //        key, bkt->keys[j], bkt->values[j]);

      if (!key_equal_(bkt->keys[j], key)) continue;
      if (bkt->values[j] == kNullRowID) continue;

      auto value = reinterpret_cast<uint64_t>(&(bkt->values[j]));
      found++;

      if (!func(bkt->keys[j], value)) return found;
      if (UniqueKey) {
        // There will be no matching key.
        return found;
      }

      // To omit checking j if the bucket has only one item.
      if (Bucket::kBucketSize == 1) break;
    }

    bkt_id = bkt->next;
    if (bkt_id == kNullRowID) {
      return found;
    }
  }
}

} // end of namespace transaction
} // end of namespace mica

#endif
