#define CONFIG_H "silo/config/config-perf.h"
#include "silo/rcu.h"
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "tatp.h"
#include "tatp_query.h"
#include "tatp_helper.h"
#include "query.h"
#include "wl.h"
#include "thread.h"
#include "table.h"
#include "row.h"
#include "index_hash.h"
#include "index_btree.h"
#include "index_mica.h"
#include "index_mbtree.h"
#include "tatp_const.h"
#include "mem_alloc.h"
#include "catalog.h"

// static void FAIL_ON_ABORT() { assert(false); }

static void FAIL_ON_ABORT() {}

void tatp_txn_man::init(thread_t* h_thd, workload* h_wl, uint64_t thd_id) {
  txn_man::init(h_thd, h_wl, thd_id);
  _wl = (tatp_wl*)h_wl;
}

RC tatp_txn_man::run_txn(base_query* query) {
  RC rc;

  auto m_query = (tatp_query*)query;
  // printf("%d\n", (int)m_query->type);
  switch (m_query->type) {
    case TATPTxnType::DeleteCallForwarding:
#if CC_ALG == MICA
      mica_tx->begin(false);
#endif
      rc = run_delete_call_forwarding(m_query);
      break;
    case TATPTxnType::GetAccessData:
#if CC_ALG == MICA
      mica_tx->begin(true);
#endif
      rc = run_get_access_data(m_query);
      break;
    case TATPTxnType::GetNewDestination:
#if CC_ALG == MICA
      mica_tx->begin(true);
#endif
      rc = run_get_new_destination(m_query);
      break;
    case TATPTxnType::GetSubscriberData:
#if CC_ALG == MICA
      mica_tx->begin(true);
#endif
      rc = run_get_subscriber_data(m_query);
      break;
    case TATPTxnType::InsertCallForwarding:
#if CC_ALG == MICA
      mica_tx->begin(false);
#endif
      rc = run_insert_call_forwarding(m_query);
      break;
    case TATPTxnType::UpdateLocation:
#if CC_ALG == MICA
      mica_tx->begin(false);
#endif
      rc = run_update_location(m_query);
      break;
    case TATPTxnType::UpdateSubscriberData:
#if CC_ALG == MICA
      mica_tx->begin(false);
#endif
      rc = run_update_subscriber_data(m_query);
      break;
    default:
      rc = ERROR;
      assert(false);
  }

  return rc;
}

bool tatp_txn_man::get_sub_id(const char* sub_nbr, uint32_t* out_s_id) {
  auto index = _wl->i_subscriber_sub_nbr;
  auto key = subscriberSubNbrKey(sub_nbr);
  auto part_id = key_to_part(key);
  auto row = search(index, key, part_id, RD);
  if (row == NULL) return false;
  row->get_value((int)SubscriberConst::s_id, *out_s_id);
  return true;
}

row_t* tatp_txn_man::get_special_facility(uint32_t s_id, uint8_t sf_type) {
  auto index = _wl->i_special_facility;
  auto key = specialFacilityKey(s_id, sf_type);
  auto part_id = key_to_part(s_id);
  return search(index, key, part_id, RD);
}

template <typename Func>
bool tatp_txn_man::enum_call_forwarding(uint32_t s_id, uint8_t sf_type,
                                        const Func& func) {
  auto index = _wl->i_call_forwarding;
  auto key = callForwardingKey(s_id, sf_type, 0);
  auto max_key = callForwardingKey(s_id, sf_type, 23);
  auto part_id = key_to_part(s_id);

  uint64_t cnt = 4;
  row_t* rows[4];
  auto idx_rc = index_read_range(index, key, max_key, rows, cnt, part_id);
  if (idx_rc == Abort) return false;
  assert(cnt < 4);

  for (uint64_t i = 0; i < cnt; i++) {
    auto shared = rows[i];
    auto local = get_row(index, shared, part_id, RD);
    if (local == NULL) return false;

    if (!func(local)) break;
  }
  return true;
}

RC tatp_txn_man::run_delete_call_forwarding(tatp_query* query) {
  auto& arg = query->args.delete_call_forwarding;

  uint32_t s_id;
  if (!get_sub_id(arg.sub_nbr, &s_id)) {
    // printf("1\n");
    FAIL_ON_ABORT();
    return finish(Abort);
  }

  auto index = _wl->i_call_forwarding;
  auto key = callForwardingKey(s_id, arg.sf_type, 0);
  auto max_key = callForwardingKey(s_id, arg.sf_type, 23);
  auto part_id = key_to_part(s_id);

  uint64_t cnt = 4;
  row_t* rows[4];
  auto idx_rc = index_read_range(index, key, max_key, rows, cnt, part_id);
  if (idx_rc == Abort) {
    // printf("2\n");
    FAIL_ON_ABORT();
    return finish(Abort);
  }
  assert(cnt < 4);

  for (uint64_t i = 0; i < cnt; i++) {
#if CC_ALG != MICA
    auto shared = rows[i];
    // auto local = get_row(index, shared, part_id, RD);
    // DBx1000 cannot handle upgrades, so we have to request WR.
    auto local = get_row(index, shared, part_id, WR);
    if (local == NULL) {
      // printf("2-1\n");
      FAIL_ON_ABORT();
      return finish(Abort);
    }

    uint8_t start_time;
    local->get_value((int)CallForwardingConst::start_time, start_time);
    if (start_time != arg.start_time) continue;

    // Get a new local row with WR to allow detecting conflicts.
    // local = get_row(index, shared, part_id, WR);
    if (!remove_row(shared)) {
      // printf("3\n");
      FAIL_ON_ABORT();
      return finish(Abort);
    }

#else  // CC_ALG == MICA

    // MICA handles row deletion directly without using remove_row().
    auto table = _wl->t_call_forwarding;
    auto row_id = reinterpret_cast<uint64_t>(rows[i]);
    MICARowAccessHandle rah(mica_tx);
    if (!rah.peek_row(table->mica_tbl[part_id], 0, row_id, false, true, true) ||
        !rah.read_row()) {
      // printf("4\n");
      FAIL_ON_ABORT();
      return finish(Abort);
    }

    row_t tmp_row;
    auto row = &tmp_row;
    row->table = table;
#if !TPCC_CF
    row->data = const_cast<char*>(rah.cdata());
#else
    row->cf_data[0] = const_cast<char*>(rah.cdata());
#endif

    uint8_t start_time;
    row->get_value((int)CallForwardingConst::start_time, start_time);
    if (start_time != arg.start_time) continue;

    if (!rah.write_row(0) || !rah.delete_row()) {
      // printf("5\n");
      FAIL_ON_ABORT();
      return finish(Abort);
    }
#endif

    auto key = callForwardingKey(s_id, arg.sf_type, start_time);
    if (!remove_idx(index, key, rows[i], part_id)) {
      // printf("6\n");
      FAIL_ON_ABORT();
      return finish(Abort);
    }
  }
  // printf("7\n");
  return finish(RCOK);
}

RC tatp_txn_man::run_get_access_data(tatp_query* query) {
  auto& arg = query->args.get_access_data;

  auto index = _wl->i_access_info;
  auto key = accessInfoKey(arg.s_id, arg.ai_type);
  auto part_id = key_to_part(arg.s_id);
  auto row = search(index, key, part_id, RD);
  if (row == NULL) {
    // This may return no matching row.
    return finish(RCOK);
  }

  uint16_t data1;
  uint16_t data2;
  char data3[3];
  char data4[5];
  row->get_value((int)AccessInfoConst::data1, data1);
  row->get_value((int)AccessInfoConst::data2, data2);
  memcpy(data3, row->get_value((int)AccessInfoConst::data3), 3);
  memcpy(data4, row->get_value((int)AccessInfoConst::data4), 5);
  return finish(RCOK);
}

RC tatp_txn_man::run_get_new_destination(tatp_query* query) {
  auto& arg = query->args.get_new_destination;

  auto row = get_special_facility(arg.s_id, arg.sf_type);
  if (row == NULL) {
    // This may return no matching row, which is acceptable.
    return finish(RCOK);
  }
  uint8_t is_active;
  row->get_value((int)SpecialFacilityConst::is_active, is_active);
  if (!is_active) {
    // This special facility may not be active.  We accept it.
    return finish(RCOK);
  }

  if (!enum_call_forwarding(arg.s_id, arg.sf_type, [&arg](auto& row) {
        uint8_t start_time;
        row->get_value((int)CallForwardingConst::start_time, start_time);
        if (start_time > arg.start_time) return true;

        uint8_t end_time;
        row->get_value((int)CallForwardingConst::end_time, end_time);
        if (end_time <= arg.end_time) return true;

        char numberx[TATP_SUB_NBR_PADDING_SIZE];
        memcpy(numberx, row->get_value((int)CallForwardingConst::numberx),
               TATP_SUB_NBR_PADDING_SIZE);
        return true;
      })) {
    FAIL_ON_ABORT();
    return finish(Abort);
  }

  return finish(RCOK);
}

RC tatp_txn_man::run_get_subscriber_data(tatp_query* query) {
  auto& arg = query->args.get_subscriber_data;

  auto index = _wl->i_subscriber;
  auto key = subscriberKey(arg.s_id);
  auto part_id = key_to_part(arg.s_id);
  auto row = search(index, key, part_id, RD);
  if (row == NULL) {
    FAIL_ON_ABORT();
    return finish(Abort);
  }

  char data[_wl->t_subscriber->get_schema()->get_tuple_size()];
  memcpy(data, row->get_value(0), sizeof(data));
  return finish(RCOK);
}

RC tatp_txn_man::run_insert_call_forwarding(tatp_query* query) {
  auto& arg = query->args.insert_call_forwarding;

  uint32_t s_id;
  if (!get_sub_id(arg.sub_nbr, &s_id)) {
    FAIL_ON_ABORT();
    return finish(Abort);
  }

  bool valid_foreign_key = false;
  static const uint64_t spe_arr[] = {1, 2, 3, 4};
  // We need to scan (instead of point-query) due to the specification.
  for (size_t i = 0; i < sizeof(spe_arr) / sizeof(spe_arr[0]); i++) {
    auto row = get_special_facility(s_id, spe_arr[i]);
    if (row != NULL && spe_arr[i] == arg.sf_type) {
      valid_foreign_key = true;
      break;
    }
  }
  if (!valid_foreign_key) {
    // Foreign key failre -- accept it without aborting TX.
    return finish(RCOK);
  }

  // Check invariants.
  // Note that the invariant may be broken after this checking, but it will be handled by the concurrency control.
  bool unique = true;
  if (!enum_call_forwarding(s_id, arg.sf_type, [&arg, &unique](auto& row) {
        uint8_t start_time;
        row->get_value((int)CallForwardingConst::start_time, start_time);
        // printf("%d\n", start_time);
        if (start_time != arg.start_time) return true;
        unique = false;
        return false;
      })) {
    FAIL_ON_ABORT();
    return finish(Abort);
  }

  if (!unique) {
    // Unique key failure -- accept it without aborting TX.
    return finish(RCOK);
  }

// Insert a new row.
#if CC_ALG != MICA
  row_t* row = NULL;
#else
  row_t row_stub;
  auto row = &row_stub;
#endif
  uint64_t row_id;
  auto part_id = key_to_part(s_id);
  if (!insert_row(_wl->t_call_forwarding, row, part_id, row_id)) {
    FAIL_ON_ABORT();
    return finish(Abort);
  }
  int col = 0;
  row->set_value(col++, s_id);
  row->set_value(col++, arg.sf_type);
  row->set_value(col++, arg.start_time);
  row->set_value(col++, arg.end_time);
  row->set_value(col++, arg.numberx);

  auto idx = _wl->i_call_forwarding;
  auto key = callForwardingKey(s_id, arg.sf_type, arg.start_time);
  if (!insert_idx(idx, key, row, part_id)) {
    FAIL_ON_ABORT();
    return finish(Abort);
  };

  return finish(RCOK);
}

RC tatp_txn_man::run_update_location(tatp_query* query) {
  auto& arg = query->args.update_location;

  auto index = _wl->i_subscriber_sub_nbr;
  auto key = subscriberSubNbrKey(arg.sub_nbr);
  auto part_id = key_to_part(key);
  auto row = search(index, key, part_id, WR);
  if (row == NULL) {
    // There must be a matching subscriber.
    FAIL_ON_ABORT();
    return finish(Abort);
  }
  // uint32_t s_id;
  // row->get_value((int)SubscriberConst::s_id, s_id);

  row->set_value((int)SubscriberConst::vlr_location, arg.vlr_location);
  return finish(RCOK);
}

RC tatp_txn_man::run_update_subscriber_data(tatp_query* query) {
  auto& arg = query->args.update_subscriber_data;

  {
    auto index = _wl->i_subscriber;
    auto key = subscriberKey(arg.s_id);
    auto part_id = key_to_part(arg.s_id);
    auto row = search(index, key, part_id, WR);
    if (row == NULL) {
      // There must be a subscriber row.
      FAIL_ON_ABORT();
      return finish(Abort);
    }

    row->set_value((int)SubscriberConst::bit_1, arg.bit_1);
  }

  {
    auto index = _wl->i_special_facility;
    auto key = specialFacilityKey(arg.s_id, arg.sf_type);
    auto part_id = key_to_part(arg.s_id);

    row_t* row;
    auto rc = index_read(index, key, &row, part_id);
    if (rc == Abort) {
      FAIL_ON_ABORT();
      return finish(Abort);
    }
    // It is OK if there is no row.
    if (rc == ERROR) return finish(RCOK);

    auto local = get_row(index, row, part_id, WR);

    // We still should be able to write to an existing row.
    if (local == NULL) {
      FAIL_ON_ABORT();
      return finish(Abort);
    }

    local->set_value((int)SpecialFacilityConst::data_a, arg.data_a);
  }

  return finish(RCOK);
}
