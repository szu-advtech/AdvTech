#pragma once
#include "global.h"
#include "helper.h"

// borrow some functions from TPCC
#include "tpcc_helper.h"

// SUBSCRIBER_IDX
uint64_t subscriberKey(uint64_t s_id);

// SUBSCRIBER_SUB_NBR_IDX
uint64_t subscriberSubNbrKey(const char* sub_nbr);

// ACCESS_INFO_IDX
uint64_t accessInfoKey(uint64_t s_id, uint64_t ai_type);

// SPECIAL_FACILITY_IDX
uint64_t specialFacilityKey(uint64_t s_id, uint64_t sf_type);

// CALL_FORWARDING_IDX
uint64_t callForwardingKey(uint64_t s_id, uint64_t sf_type,
                           uint64_t start_time);

bool tatp_isActive(uint64_t thd_id);
uint64_t tatp_getSubscriberId(uint64_t thd_id);

uint64_t tatp_astring(int min, int max, char* str, uint64_t thd_id);
uint64_t tatp_nstring(int min, int max, char* str, uint64_t thd_id);
uint64_t tatp_randomString(int min, int max, const char* base, uint64_t numChar,
                           char* str, uint64_t thd_id);

void tatp_padWithZero(uint64_t n, char* str);

uint64_t tatp_subArr(int min, int max, const uint64_t* arr, uint64_t arrLen,
                     uint64_t* out_arr, uint64_t thd_id);
