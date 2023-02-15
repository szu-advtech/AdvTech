#include "tpcc_helper.h"
#include </home/ls/pre-ordering/cityhash/include/city.h>

drand48_data** tpcc_buffer;
uint64_t C_255, C_1023, C_8191;

uint64_t itemKey(uint64_t i_id) { return i_id; }

uint64_t warehouseKey(uint64_t w_id) { return w_id; }

uint64_t distKey(uint64_t d_id, uint64_t d_w_id) {
  return d_w_id * DIST_PER_WARE + d_id;
}

uint64_t custKey(uint64_t c_id, uint64_t c_d_id, uint64_t c_w_id) {
  return distKey(c_d_id, c_w_id) * g_cust_per_dist + c_id;
}

uint64_t custNPKey(uint64_t c_d_id, uint64_t c_w_id, const char* c_last) {
  return CityHash64(c_last, strlen(c_last)) * g_num_wh * DIST_PER_WARE +
         distKey(c_d_id, c_w_id);
}

uint64_t stockKey(uint64_t s_i_id, uint64_t s_w_id) {
  return s_w_id * g_max_items + s_i_id;
}

uint64_t orderKey(int64_t o_id, uint64_t o_d_id, uint64_t o_w_id) {
  // Use negative o_id to allow reusing the current index interface.
  return distKey(o_d_id, o_w_id) * g_max_orderline + (g_max_orderline - o_id);
}

uint64_t orderCustKey(int64_t o_id, uint64_t o_c_id, uint64_t o_d_id,
                      uint64_t o_w_id) {
  // Use negative o_id to allow reusing the current index interface.
  return distKey(o_d_id, o_w_id) * g_cust_per_dist * g_max_orderline +
         o_c_id * g_max_orderline + (g_max_orderline - o_id);
}

uint64_t neworderKey(int64_t o_id, uint64_t o_d_id, uint64_t o_w_id) {
  return distKey(o_d_id, o_w_id) * g_max_orderline + (g_max_orderline - o_id);
}

uint64_t orderlineKey(uint64_t ol_number, int64_t ol_o_id, uint64_t ol_d_id,
                      uint64_t ol_w_id) {
  // Use negative ol_o_id to allow reusing the current index interface.
  return distKey(ol_d_id, ol_w_id) * g_max_orderline * 15 +
         (g_max_orderline - ol_o_id) * 15 + ol_number;
}

uint64_t Lastname(uint64_t num, char* name) {
  static const char* n[] = {"BAR", "OUGHT", "ABLE",  "PRI",   "PRES",
                            "ESE", "ANTI",  "CALLY", "ATION", "EING"};
  strcpy(name, n[num / 100]);
  strcat(name, n[(num / 10) % 10]);
  strcat(name, n[num % 10]);
  return strlen(name);
}

uint64_t RAND(uint64_t max, uint64_t thd_id) {
  int64_t rint64 = 0;
  lrand48_r(tpcc_buffer[thd_id], &rint64);
  return rint64 % max;
}

uint64_t URand(uint64_t x, uint64_t y, uint64_t thd_id) {
  return x + RAND(y - x + 1, thd_id);
}

void InitNURand(uint64_t thd_id) {
  C_255 = (uint64_t)URand(0, 255, thd_id);
  C_1023 = (uint64_t)URand(0, 1023, thd_id);
  C_8191 = (uint64_t)URand(0, 8191, thd_id);
}

uint64_t NURand(uint64_t A, uint64_t x, uint64_t y, uint64_t thd_id) {
  int C = 0;
  switch (A) {
    case 255:
      C = C_255;
      break;
    case 1023:
      C = C_1023;
      break;
    case 8191:
      C = C_8191;
      break;
    default:
      M_ASSERT(false, "Error! NURand\n");
      exit(-1);
  }
  return (((URand(0, A, thd_id) | URand(x, y, thd_id)) + C) % (y - x + 1)) + x;
}

uint64_t MakeAlphaString(int min, int max, char* str, uint64_t thd_id) {
  char char_list[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b',
                      'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z'};
  uint64_t cnt = URand(min, max, thd_id);
  for (uint32_t i = 0; i < cnt; i++) str[i] = char_list[URand(0L, 60L, thd_id)];
  for (int i = cnt; i < max; i++) str[i] = '\0';

  return cnt;
}

uint64_t MakeNumberString(int min, int max, char* str, uint64_t thd_id) {
  uint64_t cnt = URand(min, max, thd_id);
  for (UInt32 i = 0; i < cnt; i++) {
    uint64_t r = URand(0L, 9L, thd_id);
    str[i] = '0' + r;
  }
  return cnt;
}

uint64_t wh_to_part(uint64_t wid) {
  //All Warehouse in One Part, One Part Index for DB
  assert(wid >= 1);
  assert(g_part_cnt <= g_num_wh);
  return (wid - 1) % g_part_cnt;
  /**/
  // return 0;
}
