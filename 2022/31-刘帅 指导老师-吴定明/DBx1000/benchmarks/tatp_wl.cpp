#include "global.h"
#include "helper.h"
#include "tatp.h"
#include "wl.h"
#include "thread.h"
#include "table.h"
#include "index_hash.h"
#include "index_btree.h"
#include "index_mica.h"
#include "index_mbtree.h"
#include "tatp_helper.h"
#include "row.h"
#include "query.h"
#include "txn.h"
#include "mem_alloc.h"
#include "tatp_const.h"

RC tatp_wl::init() {
  workload::init();
  next_tid = 0;
  char* cpath = getenv("GRAPHITE_HOME");
  string path;
  if (cpath == NULL)
    path = "./benchmarks/TATP_schema.txt";
  else {
    path = string(cpath);
    path += "/tests/apps/dbms/TATP_schema.txt";
  }
  init_schema(path);

  init_table();
  return RCOK;
}

RC tatp_wl::init_schema(string schema_file) {
  workload::init_schema(schema_file);

  t_subscriber = tables["SUBSCRIBER"];
  t_access_info = tables["ACCESS_INFO"];
  t_special_facility = tables["SPECIAL_FACILITY"];
  t_call_forwarding = tables["CALL_FORWARDING"];

  i_subscriber = hash_indexes["HASH_SUBSCRIBER_IDX"];
  i_subscriber_sub_nbr = hash_indexes["HASH_SUBSCRIBER_SUB_NBR_IDX"];
  i_access_info = hash_indexes["HASH_ACCESS_INFO_IDX"];
  i_special_facility = hash_indexes["HASH_SPECIAL_FACILITY_IDX"];
  i_call_forwarding = ordered_indexes["ORDERED_CALL_FORWARDING_IDX"];

  return RCOK;
}

RC tatp_wl::init_table() {
  assert(g_init_parallelism <= g_thread_cnt);

  tpcc_buffer = new drand48_data*[g_init_parallelism];

  for (uint32_t i = 0; i < g_init_parallelism; i++) {
    tpcc_buffer[i] =
        (drand48_data*)mem_allocator.alloc(sizeof(drand48_data), -1);
    srand48_r(i + 1, tpcc_buffer[i]);
  }
  InitNURand(0);

  pthread_t* p_thds = new pthread_t[g_init_parallelism - 1];
  for (uint32_t i = 0; i < g_init_parallelism; i++) tid_lock[i] = 0;
  for (uint32_t i = 0; i < g_init_parallelism - 1; i++) {
    pthread_create(&p_thds[i], NULL, threadInitTable, this);
  }
  threadInitTable(this);
  for (uint32_t i = 0; i < g_init_parallelism - 1; i++)
    pthread_join(p_thds[i], NULL);

  printf("TAPP Data Initialization Complete!\n");
  return RCOK;
}

int tatp_wl::key_to_part(uint64_t key) { return key % g_part_cnt; }

void tatp_wl::gen_subscriber(uint64_t s_id, uint64_t thd_id) {
  row_t* new_row = NULL;
#if CC_ALG == MICA
  row_t row_container;
  new_row = &row_container;
#endif
  uint64_t row_id;
  int part_id = key_to_part(s_id);
  auto rc = t_subscriber->get_new_row(new_row, part_id, row_id);
  assert(rc == RCOK);

  int col = 0;
  new_row->set_value(col++, (uint32_t)s_id);

  char sub_nbr[TATP_SUB_NBR_PADDING_SIZE];
  tatp_padWithZero(s_id, sub_nbr);
  new_row->set_value(col++, sub_nbr);

  // BIT_##
  for (int j = 0; j < 10; j++)
    new_row->set_value(col++, (uint8_t)URand(0, 1, thd_id));

  // HEX_##
  for (int j = 0; j < 10; j++)
    new_row->set_value(col++, (uint8_t)URand(0, 15, thd_id));

  // BYTE2_##
  new_row->set_value(col++, (uint16_t)URand(0, 255, thd_id));

  // msc_location + vlr_location
  for (int j = 0; j < 2; j++)
    new_row->set_value(col++, (uint32_t)URand(0, (uint32_t)-1, thd_id));

  {
    uint64_t idx_key = subscriberKey(s_id);
    index_insert(i_subscriber, idx_key, new_row, part_id);
  }
  {
    uint64_t idx_key = subscriberSubNbrKey(sub_nbr);
    auto part_id = key_to_part(
        idx_key);  // The only index that does not use s_id for partionioning
    index_insert(i_subscriber_sub_nbr, idx_key, new_row, part_id);
    assert(rc == RCOK);
  }
}

void tatp_wl::gen_access_info(uint64_t s_id, uint64_t thd_id) {
  static const uint64_t arr[] = {1, 2, 3, 4};

  uint64_t ai_types[4];
  uint64_t len =
      tatp_subArr(1, 4, arr, sizeof(arr) / sizeof(arr[0]), ai_types, thd_id);

  for (uint64_t i = 0; i < len; i++) {
    row_t* new_row = NULL;
#if CC_ALG == MICA
    row_t row_container;
    new_row = &row_container;
#endif
    uint64_t row_id;
    int part_id = key_to_part(s_id);
    auto rc = t_access_info->get_new_row(new_row, part_id, row_id);
    assert(rc == RCOK);

    int col = 0;
    new_row->set_value(col++, (uint32_t)s_id);
    uint64_t ai_type = ai_types[i];
    new_row->set_value(col++, (uint8_t)ai_type);
    new_row->set_value(col++, (uint16_t)URand(0, 255, thd_id));
    new_row->set_value(col++, (uint16_t)URand(0, 255, thd_id));
    char data3[3];
    tatp_astring(3, 3, data3, thd_id);
    new_row->set_value(col++, data3);
    char data4[5];
    tatp_astring(5, 5, data4, thd_id);
    new_row->set_value(col++, data4);

    uint64_t idx_key = accessInfoKey(s_id, ai_type);
    index_insert(i_access_info, idx_key, new_row, part_id);
    assert(rc == RCOK);
  }
}

void tatp_wl::gen_spe_and_cal(uint64_t s_id, uint64_t thd_id) {
  static const uint64_t spe_arr[] = {1, 2, 3, 4};
  static const uint64_t cal_arr[] = {0, 8, 16};

  uint64_t sf_types[4];
  uint64_t sf_len = tatp_subArr(
      1, 4, spe_arr, sizeof(spe_arr) / sizeof(spe_arr[0]), sf_types, thd_id);

  for (uint64_t i = 0; i < sf_len; i++) {
    // special_facility
    row_t* new_row = NULL;
#if CC_ALG == MICA
    row_t row_container;
    new_row = &row_container;
#endif
    uint64_t row_id;
    int part_id = key_to_part(s_id);
    auto rc = t_special_facility->get_new_row(new_row, part_id, row_id);
    assert(rc == RCOK);

    int col = 0;
    new_row->set_value(col++, (uint32_t)s_id);
    uint64_t sf_type = sf_types[i];
    new_row->set_value(col++, (uint8_t)sf_type);
    new_row->set_value(col++, (uint8_t)tatp_isActive(thd_id));
    new_row->set_value(col++, (uint16_t)URand(0, 255, thd_id));
    new_row->set_value(col++, (uint16_t)URand(0, 255, thd_id));
    char data_b[5];
    tatp_astring(5, 5, data_b, thd_id);
    new_row->set_value(col++, data_b);

    uint64_t idx_key = specialFacilityKey(s_id, sf_type);
    index_insert(i_special_facility, idx_key, new_row, part_id);
    assert(rc == RCOK);

    // call_forwarding
    uint64_t start_times[3];
    uint64_t st_len =
        tatp_subArr(0, 3, cal_arr, sizeof(cal_arr) / sizeof(cal_arr[0]),
                    start_times, thd_id);

    for (uint64_t j = 0; j < st_len; j++) {
      row_t* new_row = NULL;
#if CC_ALG == MICA
      row_t row_container;
      new_row = &row_container;
#endif
      uint64_t row_id;
      int part_id = key_to_part(s_id);
      auto rc = t_call_forwarding->get_new_row(new_row, part_id, row_id);
      assert(rc == RCOK);

      int col = 0;
      new_row->set_value(col++, (uint32_t)s_id);
      new_row->set_value(col++, (uint8_t)sf_type);
      uint64_t start_time = start_times[j];
      new_row->set_value(col++, (uint8_t)start_time);
      new_row->set_value(col++, (uint8_t)(start_time + URand(1, 8, thd_id)));
      char numberx[TATP_SUB_NBR_PADDING_SIZE];
      tatp_nstring(TATP_SUB_NBR_PADDING_SIZE, TATP_SUB_NBR_PADDING_SIZE,
                   numberx, thd_id);
      new_row->set_value(col++, numberx);

      {
        uint64_t idx_key = callForwardingKey(s_id, sf_type, start_time);
        index_insert(i_call_forwarding, idx_key, new_row, part_id);
      }
    }
  }
}

RC tatp_wl::get_txn_man(txn_man*& txn_manager, thread_t* h_thd) {
  txn_manager = (tatp_txn_man*)mem_allocator.alloc(sizeof(tatp_txn_man),
                                                   h_thd->get_thd_id());
  new (txn_manager) tatp_txn_man();
  txn_manager->init(h_thd, this, h_thd->get_thd_id());
  return RCOK;
}

void* tatp_wl::threadInitTable(void* This) {
  tatp_wl* wl = (tatp_wl*)This;
  int tid = ATOM_FETCH_ADD(wl->next_tid, 1);
  assert(tid < (int)g_thread_cnt);

#if CC_ALG == MICA
  ::mica::util::lcore.pin_thread(tid);
  while (__sync_lock_test_and_set(&wl->tid_lock[tid], 1) == 1) usleep(100);
  wl->mica_db->activate(static_cast<uint16_t>(tid));
#else
  set_affinity(tid);
#endif

  mem_allocator.register_thread(tid);

  uint64_t slice_size = (g_sub_size + g_init_parallelism) / g_init_parallelism;
  uint64_t slice_offset = slice_size * tid;
  if (slice_offset + slice_size >= g_sub_size)
    slice_size = g_sub_size - slice_offset;
  for (uint64_t s_id = slice_offset; s_id < slice_offset + slice_size; s_id++) {
    wl->gen_subscriber(s_id, tid);
    wl->gen_access_info(s_id, tid);
    wl->gen_spe_and_cal(s_id, tid);
  }

#if CC_ALG == MICA
  wl->mica_db->deactivate(static_cast<uint16_t>(tid));
  __sync_lock_release(&wl->tid_lock[tid]);
#endif
  return NULL;
}
