#ifndef _TATP_QUERY_H_
#define _TATP_QUERY_H_

#include "global.h"
#include "helper.h"
#include "query.h"

class workload;

struct tatp_query_delete_call_forwarding {
  char sub_nbr[TATP_SUB_NBR_PADDING_SIZE];
  uint8_t sf_type;
  uint8_t start_time;
};
struct tatp_query_get_access_data {
  uint32_t s_id;
  uint8_t ai_type;
};
struct tatp_query_get_new_destination {
  uint32_t s_id;
  uint8_t sf_type;
  uint8_t start_time;
  uint8_t end_time;
};
struct tatp_query_get_subscriber_data {
  uint32_t s_id;
};
struct tatp_query_insert_call_forwarding {
  char sub_nbr[TATP_SUB_NBR_PADDING_SIZE];
  uint8_t sf_type;
  uint8_t start_time;
  uint8_t end_time;
  char numberx[TATP_SUB_NBR_PADDING_SIZE];
};
struct tatp_query_update_location {
  uint32_t vlr_location;
  char sub_nbr[TATP_SUB_NBR_PADDING_SIZE];
};
struct tatp_query_update_subscriber_data {
  uint32_t s_id;
  uint8_t bit_1;
  uint16_t data_a;
  uint8_t sf_type;
};

enum class TATPTxnType {
  DeleteCallForwarding,
  GetAccessData,
  GetNewDestination,
  GetSubscriberData,
  InsertCallForwarding,
  UpdateLocation,
  UpdateSubscriberData,
};

class tatp_query : public base_query {
 public:
  void init(uint64_t thd_id, workload* h_wl);

  TATPTxnType type;
  union {
    tatp_query_delete_call_forwarding delete_call_forwarding;
    tatp_query_get_access_data get_access_data;
    tatp_query_get_new_destination get_new_destination;
    tatp_query_get_subscriber_data get_subscriber_data;
    tatp_query_insert_call_forwarding insert_call_forwarding;
    tatp_query_update_location update_location;
    tatp_query_update_subscriber_data update_subscriber_data;
  } args;

 private:
  void gen_delete_call_forwarding(uint64_t thd_id);
  void gen_get_access_data(uint64_t thd_id);
  void gen_get_new_destination(uint64_t thd_id);
  void gen_get_subscriber_data(uint64_t thd_id);
  void gen_insert_call_forwarding(uint64_t thd_id);
  void gen_update_location(uint64_t thd_id);
  void gen_update_subscriber_data(uint64_t thd_id);
};

#endif