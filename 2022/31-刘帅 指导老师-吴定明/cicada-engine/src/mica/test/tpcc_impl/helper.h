#pragma once

#include <cstdlib>
#include <iostream>
#include <stdint.h>
#include <city.h>
#include "global.h"


/************************************************/
// atomic operations
/************************************************/
#define ATOM_ADD(dest, value) \
	__sync_fetch_and_add(&(dest), value)
#define ATOM_SUB(dest, value) \
	__sync_fetch_and_sub(&(dest), value)
// returns true if cas is successful
#define ATOM_CAS(dest, oldval, newval) \
	__sync_bool_compare_and_swap(&(dest), oldval, newval)
#define ATOM_ADD_FETCH(dest, value) \
	__sync_add_and_fetch(&(dest), value)
#define ATOM_FETCH_ADD(dest, value) \
	__sync_fetch_and_add(&(dest), value)
#define ATOM_SUB_FETCH(dest, value) \
	__sync_sub_and_fetch(&(dest), value)

#define COMPILER_BARRIER asm volatile("" ::: "memory");
#define PAUSE { __asm__ ( "pause;" ); }
// #define PAUSE usleep(1);

/************************************************/
// ASSERT Helper
/************************************************/
#define M_ASSERT(cond, ...) \
	if (!(cond)) {\
		printf("ASSERTION FAILURE [%s : %d] ", \
		__FILE__, __LINE__); \
		printf(__VA_ARGS__);\
		assert(false);\
	}

#define ASSERT(cond) assert(cond)


/************************************************/
// STACK helper (push & pop)
/************************************************/
#define STACK_POP(stack, top) { \
	if (stack == NULL) top = NULL; \
	else {	top = stack; 	stack=stack->next; } }
#define STACK_PUSH(stack, entry) {\
	entry->next = stack; stack = entry; }

/************************************************/
// LIST helper (read from head & write to tail)
/************************************************/
#define LIST_GET_HEAD(lhead, ltail, en) {\
	en = lhead; \
	lhead = lhead->next; \
	if (lhead) lhead->prev = NULL; \
	else ltail = NULL; \
	en->next = NULL; }
#define LIST_PUT_TAIL(lhead, ltail, en) {\
	en->next = NULL; \
	en->prev = NULL; \
	if (ltail) { en->prev = ltail; ltail->next = en; ltail = en; } \
	else { lhead = en; ltail = en; }}
#define LIST_INSERT_BEFORE(entry, newentry) { \
	newentry->next = entry; \
	newentry->prev = entry->prev; \
	if (entry->prev) entry->prev->next = newentry; \
	entry->prev = newentry; }
#define LIST_REMOVE(entry) { \
	if (entry->next) entry->next->prev = entry->prev; \
	if (entry->prev) entry->prev->next = entry->next; }
#define LIST_REMOVE_HT(entry, head, tail) { \
	if (entry->next) entry->next->prev = entry->prev; \
	else { assert(entry == tail); tail = entry->prev; } \
	if (entry->prev) entry->prev->next = entry->next; \
	else { assert(entry == head); head = entry->next; } \
}

/************************************************/
// STATS helper
/************************************************/
#define INC_STATS(tid, name, value) \
	if (STATS_ENABLE) \
		stats._stats[tid]->name += value;

#define INC_STATS_ALWAYS(tid, name, value) \
	do { stats._stats[tid]->name += value; } while (0)

#define INC_TMP_STATS(tid, name, value) \
	if (STATS_ENABLE) \
		stats.tmp_stats[tid]->name += value;

#define INC_GLOB_STATS(name, value) \
	if (STATS_ENABLE) \
		stats.name += value;

/************************************************/
// malloc helper
/************************************************/
// In order to avoid false sharing, any unshared read/write array residing on the same
// cache line should be modified to be read only array with pointers to thread local data block.
// TODO. in order to have per-thread malloc, this needs to be modified !!!

#define ARR_PTR_MULTI(type, name, size, scale) \
	name = new type * [size]; \
	if (g_part_alloc || THREAD_ALLOC) { \
		for (UInt32 i = 0; i < size; i ++) {\
			UInt32 padsize = sizeof(type) * (scale); \
			if (g_mem_pad && padsize % CL_SIZE != 0) \
				padsize += CL_SIZE - padsize % CL_SIZE; \
			name[i] = (type *) mem_allocator.alloc(padsize, i); \
			for (UInt32 j = 0; j < scale; j++) \
				new (&name[i][j]) type(); \
		}\
	} else { \
		for (UInt32 i = 0; i < size; i++) \
			name[i] = new type[scale]; \
	}

#define ARR_PTR(type, name, size) \
	ARR_PTR_MULTI(type, name, size, 1)

#define ARR_PTR_INIT(type, name, size, value) \
	name = new type * [size]; \
	if (g_part_alloc) { \
		for (UInt32 i = 0; i < size; i ++) {\
			int padsize = sizeof(type); \
			if (g_mem_pad && padsize % CL_SIZE != 0) \
				padsize += CL_SIZE - padsize % CL_SIZE; \
			name[i] = (type *) mem_allocator.alloc(padsize, i); \
			new (name[i]) type(); \
		}\
	} else \
		for (UInt32 i = 0; i < size; i++) \
			name[i] = new type; \
	for (UInt32 i = 0; i < size; i++) \
		*name[i] = value; \

enum Data_type {DT_table, DT_page, DT_row };

// TODO currently, only DR_row supported
// data item type.

class itemid_t {
public:
	itemid_t() { };
	itemid_t(Data_type type, void * loc) {
        this->type = type;
        this->location = loc;
    };
	Data_type type;
	void * location; // points to the table | page | row
	itemid_t * next;
	bool valid;
	void init();
	bool operator==(const itemid_t &other) const;
	bool operator!=(const itemid_t &other) const;
	void operator=(const itemid_t &other);
};

int get_thdid_from_txnid(uint64_t txnid);

// key_to_part() is only for ycsb
uint64_t key_to_part(uint64_t key);
uint64_t get_part_id(void * addr);
// TODO can the following two functions be merged?
uint64_t merge_idx_key(uint64_t key_cnt, uint64_t * keys);
uint64_t merge_idx_key(uint64_t key1, uint64_t key2);
uint64_t merge_idx_key(uint64_t key1, uint64_t key2, uint64_t key3);

extern double gCPUFreq;
inline uint64_t get_server_clock() {
#if defined(__i386__)
    uint64_t ret;
    __asm__ __volatile__("rdtsc" : "=A" (ret));
#elif defined(__x86_64__)
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t ret = ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
	ret = (uint64_t) ((double)ret / gCPUFreq);
#else
	timespec * tp = new timespec;
    clock_gettime(CLOCK_REALTIME, tp);
    uint64_t ret = tp->tv_sec * 1000000000 + tp->tv_nsec;
#endif
    return ret;
}

inline uint64_t get_sys_clock() {
  #if TIME_ENABLE
	return get_server_clock();
  #else
	return 0;
  #endif
}
class myrand {
public:
	void init(uint64_t seed);
	uint64_t next();
private:
	uint64_t seed;
};

inline void set_affinity(uint64_t thd_id) {
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(thd_id, &mask);
	sched_setaffinity(0, sizeof(cpu_set_t), &mask);

	return;
	/*
	// TOOD. the following mapping only works for swarm
	// which has 4-socket, 10 physical core per socket,
	// 80 threads in total with hyper-threading
	uint64_t a = thd_id % 40;
	uint64_t processor_id = a / 10 + (a % 10) * 4;
	processor_id += (thd_id / 40) * 40;

	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(processor_id, &mask);
	sched_setaffinity(0, sizeof(cpu_set_t), &mask);
	*/
}

bool itemid_t::operator==(const itemid_t &other) const {
	return (type == other.type && location == other.location);
}

bool itemid_t::operator!=(const itemid_t &other) const {
	return !(*this == other);
}

void itemid_t::operator=(const itemid_t &other){
	this->valid = other.valid;
	this->type = other.type;
	this->location = other.location;
	assert(*this == other);
	assert(this->valid);
}

void itemid_t::init() {
	valid = false;
	location = 0;
	next = NULL;
}

int get_thdid_from_txnid(uint64_t txnid, uint64_t g_thread_cnt) {
	return txnid % g_thread_cnt;
}

uint64_t get_part_id(void * addr) {
	// return ((uint64_t)addr / PAGE_SIZE) % g_part_cnt;
}

// uint64_t key_to_part(uint64_t key) {
// 	if (g_part_alloc)
// 		return key % g_part_cnt;
// 	else
// 		return 0;
// }

uint64_t merge_idx_key(uint64_t key_cnt, uint64_t * keys) {
	uint64_t len = 64 / key_cnt;
	uint64_t key = 0;
	for (uint32_t i = 0; i < len; i++) {
		assert(keys[i] < (1UL << len));
		key = (key << len) | keys[i];
	}
	return key;
}

uint64_t merge_idx_key(uint64_t key1, uint64_t key2) {
	assert(key1 < (1UL << 32) && key2 < (1UL << 32));
	return key1 << 32 | key2;
}

uint64_t merge_idx_key(uint64_t key1, uint64_t key2, uint64_t key3) {
	assert(key1 < (1 << 21) && key2 < (1 << 21) && key3 < (1 << 21));
	return key1 << 42 | key2 << 21 | key3;
}

/****************************************************/
// Global Clock!
/****************************************************/
double gCPUFreq;
/*
inline uint64_t get_server_clock() {
#if defined(__i386__)
    uint64_t ret;
    __asm__ __volatile__("rdtsc" : "=A" (ret));
#elif defined(__x86_64__)
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t ret = ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
	ret = (uint64_t) ((double)ret / gCPUFreq);
#else
	timespec * tp = new timespec;
    clock_gettime(CLOCK_REALTIME, tp);
    uint64_t ret = tp->tv_sec * 1000000000 + tp->tv_nsec;
#endif
    return ret;
}

inline uint64_t get_sys_clock() {
#ifndef NOGRAPHITE
	static volatile uint64_t fake_clock = 0;
	if (warmup_finish)
		return CarbonGetTime();   // in ns
	else {
		return ATOM_ADD_FETCH(fake_clock, 100);
	}
#else
	if (TIME_ENABLE)
		return get_server_clock();
	return 0;
#endif
}
*/
void myrand::init(uint64_t seed) {
	this->seed = seed;
}

uint64_t myrand::next() {
	seed = (seed * 1103515247UL + 12345UL) % (1UL<<63);
	return (seed / 65537) % RAND_MAX;
}

// tpcc_helper are the following

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
  for (uint32_t i = 0; i < cnt; i++) {
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

enum {
	W_ID,
	W_NAME,
	W_STREET_1,
	W_STREET_2,
	W_CITY,
	W_STATE,
	W_ZIP,
	W_TAX,
	W_YTD
};
enum {
	D_ID,
	D_W_ID,
	D_NAME,
	D_STREET_1,
	D_STREET_2,
	D_CITY,
	D_STATE,
	D_ZIP,
	D_TAX,
	D_YTD,
	D_NEXT_O_ID
};
enum {
	C_ID,
	C_D_ID,
	C_W_ID,
	C_FIRST,
	C_MIDDLE,
	C_LAST,
	C_STREET_1,
	C_STREET_2,
	C_CITY,
	C_STATE,
	C_ZIP,
	C_PHONE,
	C_SINCE,
	C_CREDIT,
	C_CREDIT_LIM,
	C_DISCOUNT,
	C_BALANCE,
	C_YTD_PAYMENT,
	C_PAYMENT_CNT,
	C_DELIVERY_CNT,
	C_DATA
};
enum {
	H_C_ID,
	H_C_D_ID,
	H_C_W_ID,
	H_D_ID,
	H_W_ID,
	H_DATE,
	H_AMOUNT,
	H_DATA
};
enum {
	NO_O_ID,
	NO_D_ID,
	NO_W_ID,
};
enum {
	O_ID,
	O_C_ID,
	O_D_ID,
	O_W_ID,
	O_ENTRY_D,
	O_CARRIER_ID,
	O_OL_CNT,
	O_ALL_LOCAL,
};
enum {
	OL_O_ID,
	OL_D_ID,
	OL_W_ID,
	OL_NUMBER,
	OL_I_ID,
	OL_SUPPLY_W_ID,
	OL_DELIVERY_D,
	OL_QUANTITY,
	OL_AMOUNT,
	OL_DIST_INFO,
};
enum {
	I_ID,
	I_IM_ID,
	I_NAME,
	I_PRICE,
	I_DATA
};
enum {
	S_I_ID,
	S_W_ID,
	S_QUANTITY,
	S_DIST_01,
	S_DIST_02,
	S_DIST_03,
	S_DIST_04,
	S_DIST_05,
	S_DIST_06,
	S_DIST_07,
	S_DIST_08,
	S_DIST_09,
	S_DIST_10,
	S_YTD,
	S_ORDER_CNT,
	S_REMOTE_CNT,
	S_DATA
};
