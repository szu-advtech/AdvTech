#include "query.h"
#include "tpcc_query.h"
#include "tpcc.h"
#include "tpcc_helper.h"
#include "mem_alloc.h"
#include "wl.h"
#include "table.h"

void tpcc_query::init(uint64_t thd_id, workload* h_wl) {
  // RNG will use thread-specific states (thd_id) because multiple threads may make a request to the same warehouse.

  double x = (double)URand(0, 99, thd_id) / 100.0;
  part_to_access =
      (uint64_t*)mem_allocator.alloc(sizeof(uint64_t) * g_part_cnt, thd_id);
#if !TPCC_FULL
  if (x < g_perc_payment)
    gen_payment(thd_id);
  else
    gen_new_order(thd_id);
#else
  if (x < 0.04)
    gen_stock_level(thd_id);
  else if (x < 0.04 + 0.04)
    gen_delivery(thd_id);
  else if (x < 0.04 + 0.04 + 0.04)
    gen_order_status(thd_id);
  else if (x < 0.04 + 0.04 + 0.04 + 0.43)
    gen_payment(thd_id);
  else
    gen_new_order(thd_id);
#endif

#if WORKLOAD == TPCC && TPCC_SPLIT_DELIVERY
	sub_query_id = 0;
	max_sub_query_id = type == TPCC_DELIVERY ? 10 : 1;
#endif
}

void tpcc_query::gen_payment(uint64_t thd_id) {
  type = TPCC_PAYMENT;
  tpcc_query_payment& arg = args.payment;

  if (FIRST_PART_LOCAL) {
    if (g_num_wh <= g_thread_cnt)
      arg.w_id = thd_id % g_num_wh + 1;
    else {
      do {
        arg.w_id = RAND((g_num_wh + g_thread_cnt - 1) / g_thread_cnt, thd_id) *
                       g_thread_cnt +
                   thd_id + 1;
      } while (arg.w_id > g_num_wh);
      assert((arg.w_id - 1) % g_thread_cnt == thd_id);
    }
  } else
    arg.w_id = URand(1, g_num_wh, thd_id);
  uint64_t part_id = wh_to_part(arg.w_id);
  part_to_access[0] = part_id;
  part_num = 1;

  arg.d_id = URand(1, DIST_PER_WARE, thd_id);
  arg.h_amount = URand(1, 5000, thd_id);
  arg.h_date = 2013;

  int x = URand(1, 100, thd_id);
  int y = URand(1, 100, thd_id);
  if (x <= 85) {
    // home warehouse
    arg.c_d_id = arg.d_id;
    arg.c_w_id = arg.w_id;
  } else {
    // remote warehouse
    arg.c_d_id = URand(1, DIST_PER_WARE, thd_id);
    if (g_num_wh > 1) {
      while ((arg.c_w_id = URand(1, g_num_wh, thd_id)) == arg.w_id) {
      }
      if (wh_to_part(arg.w_id) != wh_to_part(arg.c_w_id)) {
        part_to_access[1] = wh_to_part(arg.c_w_id);
        part_num = 2;
      }
    } else
      arg.c_w_id = arg.w_id;
  }
  if (y <= 60) {
    // by last name
    arg.by_last_name = true;
    Lastname(NURand(255, 0, 999, thd_id), arg.c_last);
  } else {
    // by cust id
    arg.by_last_name = false;
    arg.c_id = NURand(1023, 1, g_cust_per_dist, thd_id);
  }
}

void tpcc_query::gen_new_order(uint64_t thd_id) {
  type = TPCC_NEW_ORDER;
  tpcc_query_new_order& arg = args.new_order;

  if (FIRST_PART_LOCAL) {
    if (g_num_wh <= g_thread_cnt)
      arg.w_id = thd_id % g_num_wh + 1;
    else {
      do {
        arg.w_id = RAND((g_num_wh + g_thread_cnt - 1) / g_thread_cnt, thd_id) *
                       g_thread_cnt +
                   thd_id + 1;
      } while (arg.w_id > g_num_wh);
      assert((arg.w_id - 1) % g_thread_cnt == thd_id);
    }
  } else
    arg.w_id = URand(1, g_num_wh, thd_id);
  arg.d_id = URand(1, DIST_PER_WARE, thd_id);
  arg.c_id = NURand(1023, 1, g_cust_per_dist, thd_id);
  arg.ol_cnt = URand(5, 15, thd_id);
  arg.o_entry_d = 2013;
  arg.items = (Item_no*)mem_allocator.alloc(sizeof(Item_no) * arg.ol_cnt, thd_id);
  arg.all_local = true;
  part_to_access[0] = wh_to_part(arg.w_id);
  part_num = 1;

  for (UInt32 oid = 0; oid < arg.ol_cnt; oid++) {
    arg.items[oid].ol_i_id = NURand(8191, 1, g_max_items, thd_id);
    UInt32 x = URand(1, 100, thd_id);
    if (x > 1 || g_num_wh == 1)
      arg.items[oid].ol_supply_w_id = arg.w_id;
    else {
      while ((arg.items[oid].ol_supply_w_id = URand(1, g_num_wh, thd_id)) ==
             arg.w_id) {
      }
      arg.all_local = false;
    }
    arg.items[oid].ol_quantity = URand(1, 10, thd_id);
  }
  // Remove duplicate items
  for (UInt32 i = 0; i < arg.ol_cnt; i++) {
    for (UInt32 j = 0; j < i; j++) {
      if (arg.items[i].ol_i_id == arg.items[j].ol_i_id) {
        for (UInt32 k = i; k < arg.ol_cnt - 1; k++)
          arg.items[k] = arg.items[k + 1];
        arg.ol_cnt--;
        i--;
      }
    }
  }
  for (UInt32 i = 0; i < arg.ol_cnt; i++)
    for (UInt32 j = 0; j < i; j++)
      assert(arg.items[i].ol_i_id != arg.items[j].ol_i_id);
  // update part_to_access
  for (UInt32 i = 0; i < arg.ol_cnt; i++) {
    UInt32 j;
    for (j = 0; j < part_num; j++)
      if (part_to_access[j] == wh_to_part(arg.items[i].ol_supply_w_id)) break;
    if (j == part_num)  // not found! add to it.
      part_to_access[part_num++] = wh_to_part(arg.items[i].ol_supply_w_id);
  }
  // "1% of new order gives wrong itemid"
  // We cannot do this because DBx1000 cannot distinguish user aborts from conflict aborts.
  // if (URand(0, 99, thd_id) == 0)
  //   arg.items[URand(0, arg.ol_cnt - 1, thd_id)].ol_i_id = -1;
}

void tpcc_query::gen_order_status(uint64_t thd_id) {
  type = TPCC_ORDER_STATUS;
  tpcc_query_order_status& arg = args.order_status;

  if (FIRST_PART_LOCAL) {
    if (g_num_wh <= g_thread_cnt)
      arg.w_id = thd_id % g_num_wh + 1;
    else {
      do {
        arg.w_id = RAND((g_num_wh + g_thread_cnt - 1) / g_thread_cnt, thd_id) *
                       g_thread_cnt +
                   thd_id + 1;
      } while (arg.w_id > g_num_wh);
      assert((arg.w_id - 1) % g_thread_cnt == thd_id);
    }
  } else
    arg.w_id = URand(1, g_num_wh, thd_id);
  arg.d_id = URand(1, DIST_PER_WARE, thd_id);

  int y = URand(1, 100, thd_id);
  if (y <= 60) {
    // by last name
    arg.by_last_name = true;
    Lastname(NURand(255, 0, 999, thd_id), arg.c_last);
  } else {
    // by cust id
    arg.by_last_name = false;
    arg.c_id = NURand(1023, 1, g_cust_per_dist, thd_id);
  }
}

void tpcc_query::gen_stock_level(uint64_t thd_id) {
  type = TPCC_STOCK_LEVEL;
  tpcc_query_stock_level& arg = args.stock_level;

  if (FIRST_PART_LOCAL) {
    if (g_num_wh <= g_thread_cnt)
      arg.w_id = thd_id % g_num_wh + 1;
    else {
      do {
        arg.w_id = RAND((g_num_wh + g_thread_cnt - 1) / g_thread_cnt, thd_id) *
                       g_thread_cnt +
                   thd_id + 1;
      } while (arg.w_id > g_num_wh);
      assert((arg.w_id - 1) % g_thread_cnt == thd_id);
    }
  } else
    arg.w_id = URand(1, g_num_wh, thd_id);
  arg.d_id = URand(1, DIST_PER_WARE, thd_id);
  arg.threshold = URand(10, 20, thd_id);
}

void tpcc_query::gen_delivery(uint64_t thd_id) {
  type = TPCC_DELIVERY;
  tpcc_query_delivery& arg = args.delivery;

  if (FIRST_PART_LOCAL) {
    if (g_num_wh <= g_thread_cnt)
      arg.w_id = thd_id % g_num_wh + 1;
    else {
      do {
        arg.w_id = RAND((g_num_wh + g_thread_cnt - 1) / g_thread_cnt, thd_id) *
                       g_thread_cnt +
                   thd_id + 1;
      } while (arg.w_id > g_num_wh);
      assert((arg.w_id - 1) % g_thread_cnt == thd_id);
    }
  } else
    arg.w_id = URand(1, g_num_wh, thd_id);
  arg.o_carrier_id = URand(1, DIST_PER_WARE, thd_id);
  arg.ol_delivery_d = 2013;
}
