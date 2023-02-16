#include "global.h"
#include "helper.h"
#include "table.h"
#include "catalog.h"
#include "row.h"
#include "mem_alloc.h"

void table_t::init(Catalog* schema, uint64_t part_cnt) {
  this->table_name = schema->table_name;
  this->schema = schema;
  this->part_cnt = part_cnt;

  printf ("table_t:: init part_cnt=%lu \n", part_cnt);

#if CC_ALG == MICA

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    uint64_t thread_id = part_id % g_thread_cnt;
    ::mica::util::lcore.pin_thread(thread_id);

    static const uint64_t noninlined_data_sizes[] = {4096, 4096, 4096, 4096};
    uint64_t data_sizes[4];
    uint64_t cf_count = schema->cf_count;
    if (WORKLOAD == TPCC && (strcmp(table_name, "ORDER") == 0 ||
                             strcmp(table_name, "ORDER-LINE") == 0)) {
      // Disable inlining for ORDER-LINE.
      for (uint64_t i = 0; i < cf_count; i++)
        data_sizes[i] = noninlined_data_sizes[i];
    } else {
      for (uint64_t i = 0; i < cf_count; i++)
        data_sizes[i] = schema->cf_sizes[i];
    }

#if TPCC_VALIDATE_GAP
    if (WORKLOAD == TPCC && (strcmp(table_name, "NEW-ORDER") == 0)) {
      assert(cf_count == 1);
      data_sizes[1] = 16;
      cf_count++;
    }
#endif

    char buf[1024];
    int i = 0;
    while (true) {
      sprintf(buf, "%s_%d", table_name, i);

      if(WORKLOAD == YCSB) {
#if defined (test) || defined (CWBL)
        if (mica_db->create_table(buf, cf_count, data_sizes, g_synth_table_size)) break;
#elif defined (CZEN)
        if (mica_db->create_table(buf, cf_count, data_sizes, g_synth_table_size, g_synth_table_size/ZEN_CACHE_RATE)) break;
#endif
      }
      else if (WORKLOAD == TPCC) {
        UInt64 table_size = 0;
        if (strcmp(table_name, "WAREHOUSE") == 0) {
          table_size = g_num_wh;
        }
        else if (strcmp(table_name, "DISTRICT") == 0) {
          table_size = g_num_wh*DIST_PER_WARE;
        }
        else if (strcmp(table_name, "CUSTOMER") == 0) {
          table_size = g_num_wh*DIST_PER_WARE*g_cust_per_dist;
        }
        else if (strcmp(table_name, "HISTORY") == 0) {
          table_size = g_thread_cnt*1024;
        }
        else if (strcmp(table_name, "NEW-ORDER") == 0) {
          table_size = g_thread_cnt*1024;
        }
        else if (strcmp(table_name, "ORDER") == 0) {
          table_size = g_num_wh*DIST_PER_WARE*g_cust_per_dist;
        }
        else if (strcmp(table_name, "ORDER-LINE") == 0) {
          table_size = g_num_wh*DIST_PER_WARE*g_cust_per_dist;
        }
        else if (strcmp(table_name, "ITEM") == 0) {
          table_size = g_max_items*10;
        }
        else if (strcmp(table_name, "STOCK") == 0) {
          table_size = g_num_wh*g_max_items;
        }
        else {
          printf ("init table error!\n");
        }
        table_size = std::max (table_size, 1024UL);

#if defined (test) || defined (CWBL)
        if (mica_db->create_table(buf, cf_count, data_sizes, table_size)) {
#elif defined (CZEN)
        if (mica_db->create_table(buf, cf_count, data_sizes, table_size, table_size / ZEN_CACHE_RATE)) {
#endif
          // debug
          printf ("table_t:init create_table table_name=%s table_size=%u\n", table_name, table_size);
          break;
        }
      }

      i++;
    }
    auto p = mica_db->get_table(buf);
    assert(p);

    printf("tbl_name=%s part_id=%" PRIu64 " part_cnt=%" PRIu64 "\n", buf,
           part_id, part_cnt);
    printf("\n");

    mica_tbl.push_back(p);
  }
#endif
}

RC table_t::get_new_row(row_t*& row) {
  // this function is obsolete.
  assert(false);
  return RCOK;
}

// the row is not stored locally. the pointer must be maintained by index structure.
RC table_t::get_new_row(row_t*& row, uint64_t part_id, uint64_t& row_id) {
  RC rc = RCOK;

// XXX: this has a race condition; should be used just for non-critical purposes
// cur_tab_size++;

#if CC_ALG == MICA
  assert(row != NULL);
// We do not need a new row instance because MICA has it.
#else
  row = (row_t*)mem_allocator.alloc(row_t::alloc_size(this), part_id);
#endif
  rc = row->init(this, part_id, row_id);
  row->init_manager(row);

  return rc;
}
