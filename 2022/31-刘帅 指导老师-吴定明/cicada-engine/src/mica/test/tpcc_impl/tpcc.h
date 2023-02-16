#ifndef _TPCC_H_
#define _TPCC_H_

#include "wl.h"
#include "txn.h"
#include "table.h"

class tpcc_query;
class tpcc_wl : public workload {
 public:
  RC init();
  RC init_table();
  RC init_schema(const char* schema_file);

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
};

//================================================================================
//IMPLEMENT

RC tpcc_wl::init() {
  workload::init();

  string path = "./../src/mica/test/tpcc_impl/";
  path += "TPCC_full_schema.txt";
  cout << "reading schema file: " << path << endl;
  init_schema(path.c_str());
  cout << "TPCC schema initialized" << endl;
  init_table();
  next_tid = 0;

  return RCOK;
}
RC tpcc_wl::init_table() {
  num_wh = g_num_wh;

  /******** fill in data ************/
  // data filling process:
  //- item
  //- wh
  //	- stock
  // 	- dist
  //  	- cust
  //	  	- hist
  //		- order
  //		- new order
  //		- order line
  /**********************************/
  int wh_thd_max = std::max(g_num_wh, g_thread_cnt);
  tpcc_buffer = new drand48_data*[wh_thd_max];

  for (int i = 0; i < wh_thd_max; i++) {
    tpcc_buffer[i] =
        (drand48_data*) new drand48_data;
    srand48_r(i + 1, tpcc_buffer[i]);
  }
  InitNURand(0);

  pthread_t* p_thds = new pthread_t[g_num_wh - 1];

  for (uint32_t i = 0; i < g_num_wh - 1; i++) {
    pthread_create(&p_thds[i], NULL, threadInitWarehouse, this);
  }
  threadInitWarehouse(this);
  for (uint32_t i = 0; i < g_num_wh - 1; i++) pthread_join(p_thds[i], NULL);

  printf("TPCC Data Initialization Complete!\n");
  return RCOK;
}
RC tpcc_wl::init_schema(const char* schema_file) {
  workload::init_schema(schema_file);

  t_warehouse = tables["WAREHOUSE"];
  t_district = tables["DISTRICT"];
  t_customer = tables["CUSTOMER"];
  t_history = tables["HISTORY"];
  t_neworder = tables["NEW-ORDER"];
  t_order = tables["ORDER"];
  t_orderline = tables["ORDER-LINE"];
  t_item = tables["ITEM"];
  t_stock = tables["STOCK"];

  i_item = hash_indexes["HASH_ITEM_IDX"];
  i_warehouse = hash_indexes["HASH_WAREHOUSE_IDX"];
  i_district = hash_indexes["HASH_DISTRICT_IDX"];
  i_customer_id = hash_indexes["HASH_CUSTOMER_ID_IDX"];
  i_customer_last = hash_indexes["HASH_CUSTOMER_LAST_IDX"];
  i_stock = hash_indexes["HASH_STOCK_IDX"];
  i_order = ordered_indexes["ORDERED_ORDER_IDX"];
  i_order_cust = ordered_indexes["ORDERED_ORDER_CUST_IDX"];
  i_neworder = ordered_indexes["ORDERED_NEWORDER_IDX"];
  i_orderline = ordered_indexes["ORDERED_ORDERLINE_IDX"];
  return RCOK;
}
void tpcc_wl::init_tab_item() {
}
void tpcc_wl::init_tab_wh(uint32_t wid) {
}
void tpcc_wl::init_tab_dist(uint64_t w_id) {
}
void tpcc_wl::init_tab_stock(uint64_t w_id) {
}
void tpcc_wl::init_tab_cust(uint64_t d_id, uint64_t w_id) {
}
void tpcc_wl::init_tab_hist(uint64_t c_id, uint64_t d_id, uint64_t w_id) {
}
void tpcc_wl::init_tab_order(uint64_t d_id, uint64_t w_id) {
}
void tpcc_wl::init_permutation(uint64_t* perm_c_id, uint64_t wid) {
}
void* tpcc_wl::threadInitItem(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitWh(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitDist(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitStock(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitCust(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitHist(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitOrder(void* This) {
  return nullptr;
}
void* tpcc_wl::threadInitWarehouse(void* This) {
  return nullptr;
}

//================================================================================
//IMPLEMENT

void tpcc_txn_man::init(thread_t* h_thd, workload* h_wl) {
}
RC tpcc_txn_man::run_txn(base_query* query) {
  return RCOK;
}
RC tpcc_txn_man::run_payment(tpcc_query* m_query) {
  return RCOK;
}
RC tpcc_txn_man::run_new_order(tpcc_query* m_query) {
  return RCOK;
}
RC tpcc_txn_man::run_order_status(tpcc_query* query) {
  return RCOK;
}
RC tpcc_txn_man::run_delivery(tpcc_query* query) {
  return RCOK;
}
RC tpcc_txn_man::run_stock_level(tpcc_query* query) {
  return RCOK;
}

row_t* tpcc_txn_man::payment_getWarehouse(uint64_t w_id) {
  return nullptr;
}
void tpcc_txn_man::payment_updateWarehouseBalance(row_t* row, double h_amount) {

}
row_t* tpcc_txn_man::payment_getDistrict(uint64_t d_w_id, uint64_t d_id) {
  return nullptr;

}
void tpcc_txn_man::payment_updateDistrictBalance(row_t* row, double h_amount) {

}
row_t* tpcc_txn_man::payment_getCustomerByCustomerId(uint64_t w_id, uint64_t d_id,
                                       uint64_t c_id) {
  return nullptr;

}
row_t* tpcc_txn_man::payment_getCustomerByLastName(uint64_t w_id, uint64_t d_id,
                                     const char* c_last, uint64_t* out_c_id) {
  return nullptr;

}
bool tpcc_txn_man::payment_updateCustomer(row_t* row, uint64_t c_id, uint64_t c_d_id,
                            uint64_t c_w_id, uint64_t d_id, uint64_t w_id,
                            double h_amount) {
  return true;
}
bool tpcc_txn_man::payment_insertHistory(uint64_t c_id, uint64_t c_d_id, uint64_t c_w_id,
                           uint64_t d_id, uint64_t w_id, uint64_t h_date,
                           double h_amount, const char* h_data) {
  return true;
}

row_t* tpcc_txn_man::new_order_getWarehouseTaxRate(uint64_t w_id) {
  return nullptr;

}
row_t* tpcc_txn_man::new_order_getDistrict(uint64_t d_id, uint64_t d_w_id) {
  return nullptr;

}
void tpcc_txn_man::new_order_incrementNextOrderId(row_t* row, int64_t* out_o_id) {

}
row_t* tpcc_txn_man::new_order_getCustomer(uint64_t w_id, uint64_t d_id, uint64_t c_id) {
  return nullptr;

}
bool tpcc_txn_man::new_order_createOrder(int64_t o_id, uint64_t d_id, uint64_t w_id,
                           uint64_t c_id, uint64_t o_entry_d,
                           uint64_t o_carrier_id, uint64_t ol_cnt,
                           bool all_local) {
  return true;
}
bool tpcc_txn_man::new_order_createNewOrder(int64_t o_id, uint64_t d_id, uint64_t w_id) {
  return true;
}
row_t* tpcc_txn_man::new_order_getItemInfo(uint64_t ol_i_id) {
  return nullptr;

}
row_t* tpcc_txn_man::new_order_getStockInfo(uint64_t ol_i_id, uint64_t ol_supply_w_id) {
  return nullptr;

}
void tpcc_txn_man::new_order_updateStock(row_t* row, uint64_t ol_quantity, bool remote) {

}
bool tpcc_txn_man::new_order_createOrderLine(int64_t o_id, uint64_t d_id, uint64_t w_id,
                               uint64_t ol_number, uint64_t ol_i_id,
                               uint64_t ol_supply_w_id,
                               uint64_t ol_delivery_d, uint64_t ol_quantity,
                               double ol_amount, const char* ol_dist_info) {
  return true;
}
row_t* tpcc_txn_man::order_status_getCustomerByCustomerId(uint64_t w_id, uint64_t d_id,
                                            uint64_t c_id) {
  return nullptr;

}
row_t* tpcc_txn_man::order_status_getCustomerByLastName(uint64_t w_id, uint64_t d_id,
                                          const char* c_last,
                                          uint64_t* out_c_id) {
  return nullptr;

}
row_t* tpcc_txn_man::order_status_getLastOrder(uint64_t w_id, uint64_t d_id, uint64_t c_id) {
  return nullptr;

}
bool tpcc_txn_man::order_status_getOrderLines(uint64_t w_id, uint64_t d_id, int64_t o_id) {
  return true;
}
bool tpcc_txn_man::delivery_getNewOrder_deleteNewOrder(uint64_t d_id, uint64_t w_id,
                                         int64_t* out_o_id) {
  return true;
}
row_t* tpcc_txn_man::delivery_getCId(int64_t no_o_id, uint64_t d_id, uint64_t w_id) {
  return nullptr;
}
void tpcc_txn_man::delivery_updateOrders(row_t* row, uint64_t o_carrier_id) {
}
bool tpcc_txn_man::delivery_updateOrderLine_sumOLAmount(uint64_t o_entry_d, int64_t no_o_id,
                                          uint64_t d_id, uint64_t w_id,
                                          double* out_ol_total) {
  return true;
}
bool tpcc_txn_man::delivery_updateCustomer(double ol_total, uint64_t c_id, uint64_t d_id,
                             uint64_t w_id) {
  return true;
}
row_t* tpcc_txn_man::stock_level_getOId(uint64_t d_w_id, uint64_t d_id) {
  return nullptr;

}
bool tpcc_txn_man::stock_level_getStockCount(uint64_t ol_w_id, uint64_t ol_d_id,
                               int64_t ol_o_id, uint64_t s_w_id,
                               uint64_t threshold,
                               uint64_t* out_distinct_count) {
  return true;
}

#endif
