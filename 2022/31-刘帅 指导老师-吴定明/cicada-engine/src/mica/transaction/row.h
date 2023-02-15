#pragma once
#ifndef MICA_TRANSACTION_ROW_H_
#define MICA_TRANSACTION_ROW_H_

#include "mica/common.h"

namespace mica {
namespace transaction {
enum class RowVersionStatus : uint8_t {
  kInvalid = 0,
  kPending,
  kAborted,
  kCommitted,  // Commited as a valid version.
  kDeleted,    // Commited as a deleted row.
};

template <class StaticConfig>
struct PmemRowVersion {
  typename StaticConfig::Timestamp wts;
  uint64_t pkey;
  uint8_t  cmark;  // wts/pkey/cmark in a cacheline, cmark can combined with wts
  char data[0] __attribute__((aligned(8)));
} __attribute__((aligned(8)));

template <class StaticConfig>
struct RowVersion;

template <class StaticConfig>
struct RowCommon {
  RowVersion<StaticConfig>* volatile older_rv;
  PmemRowVersion<StaticConfig> *aep_rv;
};

template <class StaticConfig>
struct RowVersion : public RowCommon<StaticConfig> {
  typename StaticConfig::Timestamp wts;
  typename StaticConfig::ConcurrentTimestamp rts;

  volatile RowVersionStatus status;

  uint8_t numa_id;     // NUMA node ID (set by Table or SharedRowVersionPool).
  uint16_t size_cls;   // Size class (set by Table or SharedRowVersionPool).
  uint32_t data_size;  // Data size (set by Context).

  static constexpr uint8_t kInlinedRowVersionNUMAID = static_cast<uint8_t>(-1);
  bool is_inlined() const { return numa_id == kInlinedRowVersionNUMAID; }

  char data[0] __attribute__((aligned(8)));
};  // Alignment of Rows is handled by the row pool manually.

template <class StaticConfig>
struct RowHead : public RowCommon<StaticConfig> {
  // status  0 can be replace
  //         1 can not be replace
  //        ee thread operate on it, can not be replace
  //        ff not avaliable

  // std::atomic<uint8_t> status;
  volatile uint8_t status;  // also used for multiple-thread cas recovery
  uint64_t pkey;
  RowVersion<StaticConfig> inlined_rv[0] __attribute__((aligned(8)));
} __attribute__((aligned(8)));  // Alignment of Rows is handled by the table manually.

template <class StaticConfig>
struct RowGCInfo {
  typename StaticConfig::ConcurrentTimestamp gc_ts;
  volatile uint32_t gc_lock;
} __attribute__((aligned(8)));  // __attribute__((aligned(64)));

}
}

#endif

