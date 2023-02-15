#ifndef _TPCC_H_
#define _TPCC_H_

#include "wl.h"
#include "txn.h"

class table_t;
class INDEX;
class tpcc_query;

// #define TPCC_SILO_REF_LAST_NO_O_IDS
// #define TPCC_DBX1000_SERIAL_DELIVERY
// #define TPCC_CAVALIA_NO_OL_UPDATE
// #define TPCC_FOEDUS_DUPLICATE_ITEM
// #define EMULATE_SNAPSHOT_FOR_1VCC // Runs read-only TX in repeatable read mode

class tpcc_wl : public workload {
 public:
  RC init();
  RC init_table();
  RC init_schema(const char* schema_file);
  RC get_txn_man(txn_man*& txn_manager, thread_t* h_thd);
  table_t* t_warehouse;
  table_t* t_district;
  table_t* t_customer;
  table_t* t_history;
  table_t* t_neworder;
  table_t* t_order;
  table_t* t_orderline;
  table_t* t_item;
  table_t* t_stock;

  HASH_INDEX* i_item;
  HASH_INDEX* i_warehouse;
  HASH_INDEX* i_district;
  HASH_INDEX* i_customer_id;
  HASH_INDEX* i_customer_last;
  HASH_INDEX* i_stock;
  ORDERED_INDEX* i_order;
  ORDERED_INDEX* i_order_cust;
  ORDERED_INDEX* i_neworder;
  ORDERED_INDEX* i_orderline;

  bool** delivering;
  uint32_t next_tid;

  uint32_t tid_lock[256];

 private:
  uint64_t num_wh;
  void init_tab_item();
  void init_tab_wh(uint32_t wid);
  void init_tab_dist(uint64_t w_id);
  void init_tab_stock(uint64_t w_id);
  void init_tab_cust(uint64_t d_id, uint64_t w_id);
  void init_tab_hist(uint64_t c_id, uint64_t d_id, uint64_t w_id);
  void init_tab_order(uint64_t d_id, uint64_t w_id);

  void init_permutation(uint64_t* perm_c_id, uint64_t wid);

  static void* threadInitItem(void* This);
  static void* threadInitWh(void* This);
  static void* threadInitDist(void* This);
  static void* threadInitStock(void* This);
  static void* threadInitCust(void* This);
  static void* threadInitHist(void* This);
  static void* threadInitOrder(void* This);

  static void* threadInitWarehouse(void* This);
};

class tpcc_txn_man : public txn_man {
 public:
  void init(thread_t* h_thd, workload* h_wl, uint64_t part_id);
  RC run_txn(base_query* query);

 private:
  tpcc_wl* _wl;
  RC run_payment(tpcc_query* m_query);
  RC run_new_order(tpcc_query* m_query);
  RC run_order_status(tpcc_query* query);
  RC run_delivery(tpcc_query* query);
  RC run_stock_level(tpcc_query* query);

#ifdef TPCC_SILO_REF_LAST_NO_O_IDS
  struct LastNoOID {
    uint64_t o_id;
  } __attribute__((aligned(CL_SIZE)));
  LastNoOID last_no_o_ids[NUM_WH * DIST_PER_WARE];
#endif

#ifdef TPCC_DBX1000_SERIAL_DELIVERY
  struct ActiveDelivery {
    uint32_t lock;
  } __attribute__((aligned(CL_SIZE)));

  ActiveDelivery active_delivery[NUM_WH];
#endif

  row_t* payment_getWarehouse(uint64_t w_id);
  void payment_updateWarehouseBalance(row_t* row, double h_amount);
  row_t* payment_getDistrict(uint64_t d_w_id, uint64_t d_id);
  void payment_updateDistrictBalance(row_t* row, double h_amount);
  row_t* payment_getCustomerByCustomerId(uint64_t w_id, uint64_t d_id,
                                         uint64_t c_id);
  row_t* payment_getCustomerByLastName(uint64_t w_id, uint64_t d_id,
                                       const char* c_last, uint64_t* out_c_id);
  bool payment_updateCustomer(row_t* row, uint64_t c_id, uint64_t c_d_id,
                              uint64_t c_w_id, uint64_t d_id, uint64_t w_id,
                              double h_amount);
  bool payment_insertHistory(uint64_t c_id, uint64_t c_d_id, uint64_t c_w_id,
                             uint64_t d_id, uint64_t w_id, uint64_t h_date,
                             double h_amount, const char* h_data);

  row_t* new_order_getWarehouseTaxRate(uint64_t w_id);
  row_t* new_order_getDistrict(uint64_t d_id, uint64_t d_w_id);
  void new_order_incrementNextOrderId(row_t* row, int64_t* out_o_id);
  row_t* new_order_getCustomer(uint64_t w_id, uint64_t d_id, uint64_t c_id);
  bool new_order_createOrder(int64_t o_id, uint64_t d_id, uint64_t w_id,
                             uint64_t c_id, uint64_t o_entry_d,
                             uint64_t o_carrier_id, uint64_t ol_cnt,
                             bool all_local);
  bool new_order_createNewOrder(int64_t o_id, uint64_t d_id, uint64_t w_id);
  row_t* new_order_getItemInfo(uint64_t ol_i_id);
  row_t* new_order_getStockInfo(uint64_t ol_i_id, uint64_t ol_supply_w_id);
  void new_order_updateStock(row_t* row, uint64_t ol_quantity, bool remote);
  bool new_order_createOrderLine(int64_t o_id, uint64_t d_id, uint64_t w_id,
                                 uint64_t ol_number, uint64_t ol_i_id,
                                 uint64_t ol_supply_w_id,
                                 uint64_t ol_delivery_d, uint64_t ol_quantity,
                                 double ol_amount, const char* ol_dist_info);

  row_t* order_status_getCustomerByCustomerId(uint64_t w_id, uint64_t d_id,
                                              uint64_t c_id);
  row_t* order_status_getCustomerByLastName(uint64_t w_id, uint64_t d_id,
                                            const char* c_last,
                                            uint64_t* out_c_id);
  row_t* order_status_getLastOrder(uint64_t w_id, uint64_t d_id, uint64_t c_id);
  bool order_status_getOrderLines(uint64_t w_id, uint64_t d_id, int64_t o_id);

  bool delivery_getNewOrder_deleteNewOrder(uint64_t d_id, uint64_t w_id,
                                           int64_t* out_o_id);
  row_t* delivery_getCId(int64_t no_o_id, uint64_t d_id, uint64_t w_id);
  void delivery_updateOrders(row_t* row, uint64_t o_carrier_id);
  bool delivery_updateOrderLine_sumOLAmount(uint64_t o_entry_d, int64_t no_o_id,
                                            uint64_t d_id, uint64_t w_id,
                                            double* out_ol_total);
  bool delivery_updateCustomer(double ol_total, uint64_t c_id, uint64_t d_id,
                               uint64_t w_id);

  row_t* stock_level_getOId(uint64_t d_w_id, uint64_t d_id);
  bool stock_level_getStockCount(uint64_t ol_w_id, uint64_t ol_d_id,
                                 int64_t ol_o_id, uint64_t s_w_id,
                                 uint64_t threshold,
                                 uint64_t* out_distinct_count);

  // for CZEN usd, universal for other use if needed
  row_t* payment_getCustomerByCustomerPKey(uint64_t w_id, uint64_t key);
  row_t* order_status_getCustomerByCustomerPKey(uint64_t w_id, uint64_t key);
  row_t* order_status_getLastOrderByCustomerPKey (uint64_t w_id, uint64_t key);
};

#endif
