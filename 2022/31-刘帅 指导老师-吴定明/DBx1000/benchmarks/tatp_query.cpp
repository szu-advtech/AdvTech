#include "query.h"
#include "tatp_query.h"
#include "tatp.h"
#include "tatp_helper.h"
#include "mem_alloc.h"
#include "wl.h"
#include "table.h"

void tatp_query::init(uint64_t thd_id, workload* h_wl) {
  int64_t x = (int64_t)URand(0, 99, thd_id);
  if ((x -= TATP_FREQUENCY_DELETE_CALL_FORWARDING) < 0)
    gen_delete_call_forwarding(thd_id);
  else if ((x -= TATP_FREQUENCY_GET_ACCESS_DATA) < 0)
    gen_get_access_data(thd_id);
  else if ((x -= TATP_FREQUENCY_GET_NEW_DESTINATION) < 0)
    gen_get_new_destination(thd_id);
  else if ((x -= TATP_FREQUENCY_GET_SUBSCRIBER_DATA) < 0)
    gen_get_subscriber_data(thd_id);
  else if ((x -= TATP_FREQUENCY_INSERT_CALL_FORWARDING) < 0)
    gen_insert_call_forwarding(thd_id);
  else if ((x -= TATP_FREQUENCY_UPDATE_LOCATION) < 0)
    gen_update_location(thd_id);
  else if ((x -= TATP_FREQUENCY_UPDATE_SUBSCRIBER_DATA) < 0)
    gen_update_subscriber_data(thd_id);
  else
    assert(false);
}

void tatp_query::gen_delete_call_forwarding(uint64_t thd_id) {
  type = TATPTxnType::DeleteCallForwarding;
  auto& arg = args.delete_call_forwarding;

  uint64_t s_id = tatp_getSubscriberId(thd_id);
  tatp_padWithZero(s_id, arg.sub_nbr);
  arg.sf_type = (uint8_t)URand(1, 4, thd_id);
  arg.start_time = (uint8_t)(8 * URand(0, 2, thd_id));
}

void tatp_query::gen_get_access_data(uint64_t thd_id) {
  type = TATPTxnType::GetAccessData;
  auto& arg = args.get_access_data;

  arg.s_id = (uint32_t)tatp_getSubscriberId(thd_id);
  arg.ai_type = (uint8_t)URand(1, 4, thd_id);
}

void tatp_query::gen_get_new_destination(uint64_t thd_id) {
  type = TATPTxnType::GetNewDestination;
  auto& arg = args.get_new_destination;

  arg.s_id = (uint32_t)tatp_getSubscriberId(thd_id);
  arg.sf_type = (uint8_t)URand(1, 4, thd_id);
  arg.start_time = (uint8_t)(8 * URand(0, 2, thd_id));
  arg.end_time = (uint8_t)URand(1, 24, thd_id);
}

void tatp_query::gen_get_subscriber_data(uint64_t thd_id) {
  type = TATPTxnType::GetSubscriberData;
  auto& arg = args.get_subscriber_data;

  arg.s_id = (uint32_t)tatp_getSubscriberId(thd_id);
}

void tatp_query::gen_insert_call_forwarding(uint64_t thd_id) {
  type = TATPTxnType::InsertCallForwarding;
  auto& arg = args.insert_call_forwarding;

  uint64_t s_id = tatp_getSubscriberId(thd_id);
  tatp_padWithZero(s_id, arg.sub_nbr);
  arg.sf_type = (uint8_t)URand(1, 4, thd_id);
  arg.start_time = (uint8_t)(8 * URand(0, 2, thd_id));
  arg.end_time = (uint8_t)URand(1, 24, thd_id);
  tatp_padWithZero(s_id, arg.numberx);
}

void tatp_query::gen_update_location(uint64_t thd_id) {
  type = TATPTxnType::UpdateLocation;
  auto& arg = args.update_location;

  arg.vlr_location = (uint32_t)URand(0, (uint32_t)-1, thd_id);
  uint64_t s_id = tatp_getSubscriberId(thd_id);
  tatp_padWithZero(s_id, arg.sub_nbr);
}

void tatp_query::gen_update_subscriber_data(uint64_t thd_id) {
  type = TATPTxnType::UpdateSubscriberData;
  auto& arg = args.update_subscriber_data;

  arg.s_id = (uint32_t)tatp_getSubscriberId(thd_id);
  arg.bit_1 = (uint8_t)URand(0, 1, thd_id);
  arg.data_a = (uint16_t)URand(0, 255, thd_id);
  arg.sf_type = (uint8_t)URand(1, 4, thd_id);
}
