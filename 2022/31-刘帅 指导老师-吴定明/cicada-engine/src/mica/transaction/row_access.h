#pragma once
#ifndef MICA_TRANSACTION_TRANSACTION_ROW_ACCESS_H_
#define MICA_TRANSACTION_TRANSACTION_ROW_ACCESS_H_

#include "mica/common.h"
#include "mica/transaction/table.h"
#include "mica/transaction/row.h"
#include "mica/transaction/stats.h"
#include "mica/transaction/row_version_pool.h"

namespace mica {
namespace transaction {

// State chart:
// Initial accesses:
// new():             . -> kNew
// peek():            . -> kPeek
// Upgrades:
// read():        kPeek -> kRead
// write():       kPeek -> kWrite
// write():       kRead -> kReadWrite
// delete():     kWrite -> kDelete
// delete(): kReadWrite -> kReadDelete
// delete():       kNew -> .

enum class RowAccessState : uint8_t {
  kInvalid = 0,

  kNew,         // Has write_rv
  kPeek,        // Has read_rv
                //
  kRead,        // Has read_rv
  kReadWrite,   // Has write_rv, read_rv
  kWrite,       // Has write_rv, read_rv
  kDelete,      // Has write_rv, read_rv
  kReadDelete,  // Has write_rv, read_rv
};

template <class StaticConfig>
class Transaction;

template <class StaticConfig>
class PmemTransaction;

template <class StaticConfig>
struct RowAccessItem;

template <class StaticConfig>
class RowAccessHandle {
 public:
  Transaction<StaticConfig>* tx() { return tx_; }
  const Transaction<StaticConfig>* tx() const { return tx_; }

  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool new_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
               bool check_dup_access, uint64_t data_size,
               const DataCopier& data_copier = DataCopier()) {
    return tx_->new_row(*this, tbl, cf_id, row_id, check_dup_access, data_size,
                        data_copier);
  }
  void prefetch_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                    uint64_t off, uint64_t len) {
    tx_->prefetch_row(tbl, cf_id, row_id, off, len);
  }
  bool peek_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                bool check_dup_access, bool read_hint, bool write_hint) {
    return tx_->peek_row(*this, tbl, cf_id, row_id, check_dup_access, read_hint,
                         write_hint);
  }
  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool read_row(const DataCopier& data_copier = DataCopier()) {
    return tx_->read_row(*this, data_copier);
  }
  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool write_row(
      uint64_t data_size = Transaction<StaticConfig>::kDefaultWriteDataSize,
      const DataCopier& data_copier = DataCopier()) {
    return tx_->write_row(*this, data_size, data_copier);
  }
  bool delete_row() { return tx_->delete_row(*this); }

  RowAccessState state() const {
    if (*this)
      return access_item_->state;
    else
      return RowAccessState::kInvalid;
  }
  operator bool() const { return access_item_ != nullptr; }
  bool operator!() const { return access_item_ == nullptr; }

  Table<StaticConfig>* table() { return access_item_->tbl; }
  const Table<StaticConfig>* table() const { return access_item_->tbl; }

  uint16_t cf_id() const { return access_item_->cf_id; }

  uint64_t row_id() const { return access_item_->row_id; }

  bool can_read() const {
    return access_item_->write_rv != nullptr ||
           access_item_->read_rv != nullptr;
  }
  bool can_write() const { return access_item_->write_rv != nullptr; }
  bool is_deleted() const {
    return access_item_ == nullptr || (access_item_->write_rv != nullptr &&
                                       access_item_->write_rv->deleted) ||
           (access_item_->read_rv != nullptr && access_item_->read_rv->deleted);
  }

  const char* cdata() const {
    if (access_item_->write_rv != nullptr)
      return access_item_->write_rv->data;
    else if (access_item_->read_rv != nullptr)
      return access_item_->read_rv->data;
    else
      return nullptr;
  }

  char* data() {
    if (access_item_->write_rv != nullptr)
      return access_item_->write_rv->data;
    else
      return nullptr;
  }

  uint64_t size() const { return access_item_->tbl->data_size(); }

  uint64_t rv_size() const {
    if (access_item_->write_rv != nullptr)
      return SharedRowVersionPool<StaticConfig>::class_to_size(
          access_item_->write_rv->size_cls);
    else if (access_item_->read_rv != nullptr)
      return SharedRowVersionPool<StaticConfig>::class_to_size(
          access_item_->read_rv->size_cls);
    else
      return 0;
  }

  void reset() { access_item_ = nullptr; }

  RowAccessHandle() : access_item_(nullptr) {}

  explicit RowAccessHandle(Transaction<StaticConfig>* tx)
      : tx_(tx), access_item_(nullptr) {}

  RowAccessHandle(const RowAccessHandle& o)
      : tx_(o.tx_), access_item_(o.access_item_) {}

  RowAccessHandle& operator=(const RowAccessHandle& o) {
    tx_ = o.tx_;
    access_item_ = o.access_item_;
    return *this;
  }
  
  RowAccessItem<StaticConfig>* row_access_item () {
    return access_item_;
  }

 protected:

  friend Transaction<StaticConfig>;
  Transaction<StaticConfig>* tx_;
  RowAccessItem<StaticConfig>* access_item_;

 public:
  virtual
  bool aep_peek_row   (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access, bool read_hint,
                       bool write_hint) {
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    (void)read_hint;
    (void)write_hint;
    printf ("Not support aep_peek_row!\n");
    return false;
  }
  virtual
  bool aep_read_row   (void) {
    printf ("Not support aep_read_row!\n");
    return false;
  }
  virtual
  bool aep_write_row  (uint64_t data_size) {
    (void) data_size;
    printf ("Not support aep_write_row!\n");
    return false;
  }
  virtual
  bool aep_new_row    (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access,
                       uint64_t data_size) {
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    (void)data_size;
    printf ("Not support aep_new_row!\n");
    return false;
  }
  virtual
  bool aep_delete_row (void) {
    printf ("Not support aep_delete_row!\n");
    return false;
  }
  
  // after peek_row
  PmemRowVersion<StaticConfig>* get_aep_rv_from_peek_rv () {
    return access_item_->read_rv->aep_rv;
  }

  virtual uint64_t primary_key () {
    return access_item_ != nullptr ? 
           access_item_->head->pkey: static_cast<uint64_t>(-1);
  }

};

template <class StaticConfig>
class RowAccessHandlePeekOnly {
 public:
  Transaction<StaticConfig>* tx() { return tx_; }
  const Transaction<StaticConfig>* tx() const { return tx_; }

  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool new_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
               bool check_dup_access, uint64_t data_size,
               const DataCopier& data_copier = DataCopier()) {
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    (void)data_size;
    (void)data_copier;
    return false;
  }
  void prefetch_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                    uint64_t off, uint64_t len) {
    tx_->prefetch_row(tbl, cf_id, row_id, off, len);
  }
  bool peek_row(Table<StaticConfig>* tbl, uint16_t cf_id, uint64_t row_id,
                bool check_dup_access, bool read_hint, bool write_hint) {
    (void)read_hint;
    (void)write_hint;
    return tx_->peek_row(*this, tbl, cf_id, row_id, check_dup_access);
  }
  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool read_row(const DataCopier& data_copier = DataCopier()) {
    (void)data_copier;
    return false;
  }
  template <
      class DataCopier = typename Transaction<StaticConfig>::TrivialDataCopier>
  bool write_row(
      uint64_t data_size = Transaction<StaticConfig>::kDefaultWriteDataSize,
      const DataCopier& data_copier = DataCopier()) {
    (void)data_size;
    (void)data_copier;
    return false;
  }
  bool delete_row() { return false; }

  RowAccessState state() const {
    if (*this)
      return RowAccessState::kPeek;
    else
      return RowAccessState::kInvalid;
  }
  operator bool() const { return read_rv_ != nullptr; }
  bool operator!() const { return read_rv_ == nullptr; }

  Table<StaticConfig>* table() { return tbl_; }
  const Table<StaticConfig>* table() const { return tbl_; }

  uint16_t cf_id() const { return cf_id_; }

  uint64_t row_id() const { return row_id_; }

  bool can_read() const { return read_rv_ != nullptr; }
  bool can_write() const { return false; }
  bool is_deleted() const { return read_rv_ != nullptr && read_rv_->deleted; }

  const char* cdata() const {
    if (read_rv_ != nullptr)
      return read_rv_->data;
    else
      return nullptr;
  }

  char* data() { return nullptr; }

  size_t size() const { return tbl_->data_size(); }

  void reset() { read_rv_ = nullptr; }

  RowAccessHandlePeekOnly() : read_rv_(nullptr) {}

  explicit RowAccessHandlePeekOnly(Transaction<StaticConfig>* tx)
      : tx_(tx), tbl_(nullptr), cf_id_(0), row_id_(0), read_rv_(nullptr) {}

  RowAccessHandlePeekOnly(const RowAccessHandlePeekOnly& o)
      : tx_(o.tx_),
        tbl_(o.tbl_),
        cf_id_(o.cf_id_),
        row_id_(o.row_id_),
        read_rv_(nullptr) {}

  RowAccessHandlePeekOnly& operator=(const RowAccessHandlePeekOnly& o) {
    tx_ = o.tx_;
    tbl_ = o.tbl_;
    cf_id_ = o.cf_id_;
    row_id_ = o.row_id_;
    read_rv_ = nullptr;
    return *this;
  }

 private:
  friend Transaction<StaticConfig>;

 protected:
  Transaction<StaticConfig>* tx_;
  Table<StaticConfig>* tbl_;
  uint16_t cf_id_;
  uint64_t row_id_;
  RowVersion<StaticConfig>* read_rv_;

 public:
  virtual
  bool aep_peek_row   (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access, bool read_hint,
                       bool write_hint) {
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)check_dup_access;
    (void)read_hint;
    (void)write_hint;
    printf ("Not support aep_peek_row!\n");
    return false;
  }
  virtual
  bool aep_read_row   (void) {
    printf ("Not support aep_read_row!\n");
    return false;
  }
  virtual
  bool aep_write_row  (uint64_t data_size) {
    (void) data_size;
    printf ("Not support aep_write_row!\n");
    return false;
  }
  virtual
  bool aep_new_row    (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, 
                       uint64_t prim_key,
                       bool check_dup_access,
                       uint64_t data_size) {
    (void)tbl;
    (void)cf_id;
    (void)row_id;
    (void)prim_key;
    (void)check_dup_access;
    (void)data_size;
    printf ("Not support aep_new_row!\n");
    return false;
  }
  virtual
  bool aep_delete_row (void) {
    printf ("Not support aep_delete_row!\n");
    return false;
  }

  uint64_t primary_key () {
    return read_rv_ != nullptr ? 
           tbl_->head (cf_id_,row_id_)->pkey: static_cast<uint64_t>(-1);
  }
};

template <class StaticConfig>
struct RowAccessItem {
  // Invariant: newer_rv.wts > (write_rv.wts) > read_rv.wts.
  uint16_t i;
  uint8_t inserted;
  RowAccessState state;

  Table<StaticConfig>* tbl;
  uint16_t cf_id;
  uint64_t row_id;

  RowHead<StaticConfig>* head;
  RowCommon<StaticConfig>* newer_rv;
  RowVersion<StaticConfig>* write_rv;
  RowVersion<StaticConfig>* read_rv;
  // typename StaticConfig::Timestamp latest_wts;
};

template <class StaticConfig>
class PmemRowAccessHandle: public RowAccessHandle<StaticConfig> {
 public:
  // access all in dram
  bool aep_peek_row   (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access, bool read_hint,
                       bool write_hint) {
    return RowAccessHandle<StaticConfig>::
                              tx_->aep_peek_row (*this, 
                              tbl, cf_id, row_id, 
                              check_dup_access, read_hint, write_hint);
  }
  bool aep_new_row    (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, uint64_t prim_key,
                       bool check_dup_access, uint64_t data_size) {
    return RowAccessHandle<StaticConfig>::
                             tx_->aep_new_row (*this,
                             tbl, cf_id, row_id, prim_key,
                             check_dup_access, data_size);
  }
  bool aep_read_row   (void) {
    return RowAccessHandle<StaticConfig>::tx_->aep_read_row (*this);
  }
  bool aep_write_row  (uint64_t data_size) {
    return RowAccessHandle<StaticConfig>::tx_->aep_write_row (*this, data_size);
  }
  // must write_row before delete_row
  bool aep_delete_row (void) {
    return RowAccessHandle<StaticConfig>::tx_->aep_delete_row (*this);
  }

  PmemRowAccessHandle() : RowAccessHandle<StaticConfig>() {}

  explicit PmemRowAccessHandle(Transaction<StaticConfig>* aep_tx)
      : RowAccessHandle<StaticConfig>(aep_tx) {}

  PmemRowAccessHandle(const PmemRowAccessHandle& o)
      : RowAccessHandle<StaticConfig>(o)  {}

  PmemRowAccessHandle& operator=(const PmemRowAccessHandle &o) {
    RowAccessHandle<StaticConfig>::tx_ = o.tx_;
    RowAccessHandle<StaticConfig>::access_item_ = o.access_item_;
    return *this;
  }

 private:
  friend PmemTransaction<StaticConfig>;
};

template <class StaticConfig>
class PmemRowAccessHandlePeekOnly: public RowAccessHandlePeekOnly<StaticConfig> {
 public:
  bool aep_peek_row   (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, bool check_dup_access, bool read_hint,
                       bool write_hint) {
    (void) read_hint;
    (void) write_hint;
    return RowAccessHandlePeekOnly<StaticConfig>::
                              tx_->aep_peek_row (*this, tbl, cf_id, row_id,
                              check_dup_access);
  }

  bool aep_read_row   (void) { return false; }
  bool aep_write_row  (uint64_t data_size) { (void)data_size; 
                                              return false; }
  bool aep_new_row    (Table<StaticConfig>* tbl, uint16_t cf_id,
                       uint64_t row_id, uint64_t prim_key,
                       bool check_dup_access, uint64_t data_size) { 
    (void) tbl;
    (void) cf_id;
    (void) row_id;
    (void) prim_key;
    (void) check_dup_access;
    (void) data_size;
    return false; 
  }

  // must write_row before delete_row
  bool aep_delete_row (void) { return false; }

  PmemRowAccessHandlePeekOnly() 
      : RowAccessHandlePeekOnly<StaticConfig>(),
        aep_rv_(nullptr) {}

  explicit PmemRowAccessHandlePeekOnly(Transaction<StaticConfig>* aep_tx)
      : RowAccessHandlePeekOnly<StaticConfig>(aep_tx), 
        aep_rv_(nullptr) {}

  PmemRowAccessHandlePeekOnly(const PmemRowAccessHandlePeekOnly& o)
      : RowAccessHandlePeekOnly<StaticConfig>(), 
        RowAccessHandlePeekOnly<StaticConfig>::tx_(o.tx_),
        RowAccessHandlePeekOnly<StaticConfig>::tbl_(o.tbl_),
        RowAccessHandlePeekOnly<StaticConfig>::cf_id_(o.cf_id_),
        RowAccessHandlePeekOnly<StaticConfig>::row_id_(o.row_id_),
        RowAccessHandlePeekOnly<StaticConfig>::read_rv_(nullptr),
        RowAccessHandlePeekOnly<StaticConfig>::aep_rv_(nullptr) {}

  PmemRowAccessHandlePeekOnly& operator=(const PmemRowAccessHandlePeekOnly& o) {
    RowAccessHandlePeekOnly<StaticConfig>::tx_ = o.tx_;
    RowAccessHandlePeekOnly<StaticConfig>::tbl_ = o.tbl_;
    RowAccessHandlePeekOnly<StaticConfig>::cf_id_ = o.cf_id_;
    RowAccessHandlePeekOnly<StaticConfig>::row_id_ = o.row_id_;
    RowAccessHandlePeekOnly<StaticConfig>::read_rv_ = nullptr;
    RowAccessHandlePeekOnly<StaticConfig>::aep_rv_ = nullptr;
    return *this;
  }

  // do peek before cdata

  const char* cdata () const {
    if (aep_rv_ == nullptr) {
      return RowAccessHandlePeekOnly<StaticConfig>::cdata ();
    }
    else {
      return aep_rv_->data;
    }
  }
  bool can_read() const { 
    return RowAccessHandlePeekOnly<StaticConfig>::read_rv_ != nullptr 
           || aep_rv_ != nullptr; }
  void reset() { RowAccessHandlePeekOnly<StaticConfig>::read_rv_ = nullptr;
                  aep_rv_ = nullptr; }

  uint64_t primary_key () {
    return aep_rv_ != nullptr ? aep_rv_->pkey 
           : RowAccessHandlePeekOnly<StaticConfig>::primary_key();
  }

 private:
  friend PmemTransaction<StaticConfig>;
  PmemRowVersion<StaticConfig> *aep_rv_;
};

}
}

#endif
