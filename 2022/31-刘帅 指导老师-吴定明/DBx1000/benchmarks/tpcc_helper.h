#pragma once
#include "global.h"
#include "helper.h"

// ITEM_IDX
uint64_t itemKey(uint64_t i_id);

// WAREHOUSE_IDX
uint64_t warehouseKey(uint64_t w_id);

// DISTRICT_IDX
uint64_t distKey(uint64_t d_id, uint64_t d_w_id);

// CUSTOMER_ID_IDX
uint64_t custKey(uint64_t c_id, uint64_t c_d_id, uint64_t c_w_id);

// CUSTOMER_LAST_IDX
uint64_t custNPKey(uint64_t c_d_id, uint64_t c_w_id, const char* c_last);

// STOCK_IDX
uint64_t stockKey(uint64_t s_i_id, uint64_t s_w_id);

// ORDER_IDX
uint64_t orderKey(int64_t o_id, uint64_t o_d_id, uint64_t o_w_id);

// ORDER_CUST_IDX
uint64_t orderCustKey(int64_t o_id, uint64_t o_c_id, uint64_t o_d_id,
                      uint64_t o_w_id);

// NEWORDER_IDX
uint64_t neworderKey(int64_t o_id, uint64_t o_d_id, uint64_t o_w_id);

// ORDERED_ORDERLINE_IDX
uint64_t orderlineKey(uint64_t ol_number, int64_t ol_o_id, uint64_t ol_d_id, uint64_t ol_w_id);

uint64_t Lastname(uint64_t num, char* name);

extern drand48_data** tpcc_buffer;
extern uint64_t C_255, C_1023, C_8191;

// return random data from [0, max-1]
uint64_t RAND(uint64_t max, uint64_t thd_id);
// random number from [x, y]
uint64_t URand(uint64_t x, uint64_t y, uint64_t thd_id);
// non-uniform random number
void InitNURand(uint64_t thd_id);
uint64_t NURand(uint64_t A, uint64_t x, uint64_t y, uint64_t thd_id);
// random string with random length beteen min and max.
uint64_t MakeAlphaString(int min, int max, char* str, uint64_t thd_id);
uint64_t MakeNumberString(int min, int max, char* str, uint64_t thd_id);

uint64_t wh_to_part(uint64_t wid);
