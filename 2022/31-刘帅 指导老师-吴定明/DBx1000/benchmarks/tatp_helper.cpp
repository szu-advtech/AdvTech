#include "tatp_helper.h"
#include </home/ls/pre-ordering/cityhash/include/city.h>

uint64_t subscriberKey(uint64_t s_id) { return s_id; }

uint64_t subscriberSubNbrKey(const char* sub_nbr) {
  return CityHash64(sub_nbr, strlen(sub_nbr));
}

uint64_t accessInfoKey(uint64_t s_id, uint64_t ai_type) {
  return s_id * 4 + ai_type;
}

uint64_t specialFacilityKey(uint64_t s_id, uint64_t sf_type) {
  return s_id * 4 + sf_type;
}

uint64_t callForwardingKey(uint64_t s_id, uint64_t sf_type,
                           uint64_t start_time) {
  return s_id * 4 * 24 + sf_type * 24 + start_time;
}

uint64_t callForwardingSIDIndex(uint64_t s_id) { return s_id; }

bool tatp_isActive(uint64_t thd_id) {
  return URand(1, 100, thd_id) < URand(86, 100, thd_id) ? 1 : 0;
}

uint64_t tatp_getSubscriberId(uint64_t thd_id) {
  //return URand(1, g_sub_size, thd_id);
  if (g_sub_size <= 1000000)
    return (URand(0, 65535, thd_id) | URand(0, g_sub_size - 1, thd_id)) %
           g_sub_size;
  else if (g_sub_size <= 10000000)
    return (URand(0, 1048575, thd_id) | URand(0, g_sub_size - 1, thd_id)) %
           g_sub_size;
  else
    return (URand(0, 2097151, thd_id) | URand(0, g_sub_size - 1, thd_id)) %
           g_sub_size;
}

uint64_t tatp_astring(int min, int max, char* str, uint64_t thd_id) {
  return MakeAlphaString(min, max, str, thd_id);
}

uint64_t tatp_nstring(int min, int max, char* str, uint64_t thd_id) {
  return MakeNumberString(min, max, str, thd_id);
}

uint64_t tatp_randomString(int min, int max, const char* base, uint64_t numChar,
                           char* str, uint64_t thd_id) {
  uint64_t cnt = URand(min, max, thd_id);
  for (UInt32 i = 0; i < cnt; i++) {
    char b = base[URand(0, numChar - 1, thd_id)];
    str[i] = b;
  }
  return cnt;
}

void tatp_padWithZero(uint64_t n, char* str) {
  snprintf(str, TATP_SUB_NBR_PADDING_SIZE, "%0*" PRIu64 "",
           TATP_SUB_NBR_PADDING_SIZE, n);
}

uint64_t tatp_subArr(int min, int max, const uint64_t* arr, uint64_t arrLen,
                     uint64_t* out_arr, uint64_t thd_id) {
  uint64_t cnt = URand(min, max, thd_id);

  memcpy(out_arr, arr, sizeof(uint64_t) * arrLen);
  for (uint64_t i = 0; i < cnt; i++) {
    uint64_t j = URand(i, arrLen - 1, thd_id);

    uint64_t t = out_arr[i];
    out_arr[i] = out_arr[j];
    out_arr[j] = t;
  }
  return cnt;
}
