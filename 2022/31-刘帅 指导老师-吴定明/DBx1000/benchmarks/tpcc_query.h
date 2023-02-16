#ifndef _TPCC_QUERY_H_
#define _TPCC_QUERY_H_

#include "global.h"
#include "helper.h"
#include "query.h"

class workload;

struct tpcc_query_payment {
  uint64_t w_id;
  uint64_t d_id;
  uint64_t c_w_id;
  uint64_t c_d_id;
  double h_amount;
  uint64_t h_date;

  bool by_last_name;
  uint64_t c_id;              // by_last_name == true
  char c_last[LASTNAME_LEN];  // by_last_name == false
};

struct tpcc_query_new_order {
  uint64_t w_id;
  uint64_t d_id;
  uint64_t c_id;
  uint64_t ol_cnt;
  uint64_t o_entry_d;
  struct Item_no* items;
  bool rollback;
  bool all_local;
};

struct Item_no {
  uint64_t ol_i_id;
  uint64_t ol_supply_w_id;
  uint64_t ol_quantity;
};

struct tpcc_query_order_status {
  uint64_t w_id;
  uint64_t d_id;

  bool by_last_name;
  uint64_t c_id;              // by_last_name == true
  char c_last[LASTNAME_LEN];  // by_last_name == false
};

struct tpcc_query_stock_level {
  uint64_t w_id;
  uint64_t d_id;
  uint64_t threshold;
};

struct tpcc_query_delivery {
  uint64_t w_id;
  uint64_t o_carrier_id;
  uint64_t ol_delivery_d;
};

class tpcc_query : public base_query {
 public:
  void init(uint64_t thd_id, workload* h_wl);

  TPCCTxnType type;
  union {
    tpcc_query_payment payment;
    tpcc_query_new_order new_order;
    tpcc_query_order_status order_status;
    tpcc_query_stock_level stock_level;
    tpcc_query_delivery delivery;
  } args;

 private:
  // warehouse id to partition id mapping
  //	uint64_t wh_to_part(uint64_t wid);
  void gen_payment(uint64_t thd_id);
  void gen_new_order(uint64_t thd_id);
  void gen_order_status(uint64_t thd_id);
  void gen_stock_level(uint64_t thd_id);
  void gen_delivery(uint64_t thd_id);
};

#endif
