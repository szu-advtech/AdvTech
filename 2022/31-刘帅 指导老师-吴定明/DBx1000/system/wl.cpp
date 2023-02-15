#include "global.h"
#include "helper.h"
#include "wl.h"
#include "row.h"
#include "table.h"
#include "index_hash.h"
#include "index_array.h"
#include "index_btree.h"
#include "index_mbtree.h"
#include "index_mica.h"
#include "index_mica_mbtree.h"
#include "catalog.h"
#include "mem_alloc.h"
#include <thread>

void workload::init_mica() {
  mica_sw.init_start();
  mica_sw.init_end();

  gCPUFreq = mica_sw.c_1_sec() / 1000000000.;
  printf("detected CPU frequency = %.3f GHz\n", gCPUFreq);

#if CC_ALG == MICA
  mica_alloc = new MICAAlloc(0, MICA_ALLOC_SIZE);
  mica_aep_alloc = new MICAPmemAlloc(0, MICA_AEPALLOC_SIZE);

  // do this to malloc huge page memory or aep-memory
  mica_alloc->pre_malloc_2mb ();
  mica_aep_alloc->pre_malloc_2mb ();

  mica_page_pools[0] = new MICAPagePool(mica_alloc, mica_aep_alloc, 0);
  mica_page_pools[1] = new MICAPagePool(mica_alloc, mica_aep_alloc, 1);

  #if defined (test) || defined (CWBL)
  mica_logger = new MICAPmemLogger(mica_page_pools[1]);
  #elif defined (CZEN)
  mica_logger = new MICALogger();
  #endif

  mica_logger->disable_logging ();
  mica_db = new MICADB(mica_page_pools, mica_logger, &mica_sw, g_thread_cnt, 0);
  printf("MICA initialized\n");
#endif
}

RC workload::init() {
  sim_done = false;
  return RCOK;
}

RC workload::init_schema(string schema_file) {
#if CC_ALG == MICA
  // mica_db->activate(static_cast<uint64_t>(::mica::util::lcore.lcore_id()));
  std::vector<std::thread> threads;
  for (uint64_t thread_id = 0; thread_id < g_thread_cnt; thread_id++) {
    threads.emplace_back([&, thread_id] {
      ::mica::util::lcore.pin_thread(thread_id);
      mem_allocator.register_thread(thread_id);
      mica_db->activate(thread_id);
    });
  }
  while (threads.size() > 0) {
    threads.back().join();
    threads.pop_back();
  }
#endif

  assert(sizeof(uint64_t) == 8);
  assert(sizeof(double) == 8);
  string line;
  ifstream fin(schema_file);
  Catalog* schema;
  while (getline(fin, line)) {
    if (line.compare(0, 6, "TABLE=") == 0) {
      printf ("init_schema_table: %s\n", line.c_str());

      string tname;
      tname = &line[6];
      schema = (Catalog*)mem_allocator.alloc(sizeof(Catalog), -1);
      getline(fin, line);
      int col_count = 0;
      // Read all fields for this table.
      vector<string> lines;
      while (line.length() > 1) {
        lines.push_back(line);
        getline(fin, line);
      }
      schema->init(tname.c_str(), lines.size());
      for (UInt32 i = 0; i < lines.size(); i++) {
        string line = lines[i];
        size_t pos = 0;
        string token;
        int elem_num = 0;
        int size = 0;
        string type;
        string name;
        int cf = 0;
        while (line.length() != 0) {
          pos = line.find(",");
          if (pos == string::npos) pos = line.length();
          token = line.substr(0, pos);
          line.erase(0, pos + 1);
          switch (elem_num) {
            case 0:
              size = atoi(token.c_str());
              break;
            case 1:
              type = token;
              break;
            case 2:
              name = token;
              break;
            case 3:
#if TPCC_CF
              cf = atoi(token.c_str());
#endif
              break;
            default:
              assert(false);
          }
          elem_num++;
        }
#if WORKLOAD == YCSB
        size = MAX_TUPLE_SIZE;
#endif
        assert(elem_num == 3 || elem_num == 4);
        schema->add_col((char*)name.c_str(), size, (char*)type.c_str(), cf);
        col_count++;
      }

      int part_cnt = (CENTRAL_INDEX) ? 1 : g_part_cnt;
#if WORKLOAD == TPCC
      if (tname == "ITEM") part_cnt = 1;
#endif

#if WORKLOAD == YCSB
      assert(schema->get_tuple_size() == MAX_TUPLE_SIZE);
#endif

      table_t* cur_tab = (table_t*)mem_allocator.alloc(sizeof(table_t), -1);
      new (cur_tab) table_t;
#if CC_ALG == MICA
      cur_tab->mica_db = mica_db;
#endif
      cur_tab->init(schema, part_cnt);
      assert(schema->get_tuple_size() <= MAX_TUPLE_SIZE);
      tables[tname] = cur_tab;
    } else if (!line.compare(0, 6, "INDEX=")) {
      // debug
      // return RCOK;
      printf ("init_schema_index: index=%s", line.c_str());

      string iname;
      iname = &line[6];
      getline(fin, line);

      vector<string> items;
      string token;
      size_t pos;
      while (line.length() != 0) {
        pos = line.find(",");
        if (pos == string::npos) pos = line.length();
        token = line.substr(0, pos);
        items.push_back(token);
        line.erase(0, pos + 1);
      }

      string tname(items[0]);

      int part_cnt = (CENTRAL_INDEX) ? 1 : g_part_cnt;
#if WORKLOAD == TPCC
      if (tname == "ITEM") part_cnt = 1;
#endif

      uint64_t table_size;
#if WORKLOAD == YCSB
      table_size = g_synth_table_size;
#elif WORKLOAD == TPCC
      if (tname == "ITEM")
        table_size = stoi(items[1]);
      else
        table_size = stoi(items[1]) * g_num_wh;
#elif WORKLOAD == TATP
      table_size = stoi(items[1]) * TATP_SCALE_FACTOR;
#endif

      printf (" tname=%s\n", tname.c_str());

      if (strncmp(iname.c_str(), "ORDERED_", 8) == 0) {
        ORDERED_INDEX* index =
            (ORDERED_INDEX*)mem_allocator.alloc(sizeof(ORDERED_INDEX), -1);
        new (index) ORDERED_INDEX();

        index->init(part_cnt, tables[tname]);
        ordered_indexes[iname] = index;
      } else if (strncmp(iname.c_str(), "ARRAY_", 6) == 0) {
        ARRAY_INDEX* index = (ARRAY_INDEX*)mem_allocator.alloc(sizeof(ARRAY_INDEX), -1);
        new (index) ARRAY_INDEX();

        index->init(part_cnt, tables[tname], table_size * 2);
        array_indexes[iname] = index;
      } else if (strncmp(iname.c_str(), "HASH_", 5) == 0) {
        HASH_INDEX* index = (HASH_INDEX*)mem_allocator.alloc(sizeof(HASH_INDEX), -1);
        new (index) HASH_INDEX();

#if INDEX_STRUCT == IDX_HASH || INDEX_STRUCT == IDX_MICA
        // debug
        // printf ("part_cnt in wl.cpp = %u\n", part_cnt);
        // return RCOK;

        index->init(part_cnt, tables[tname], table_size * 2);
#else
        index->init(part_cnt, tables[tname]);
#endif
        hash_indexes[iname] = index;
      }
      else {
        printf("unrecognized index type for %s\n", iname.c_str());
        assert(false);
      }
     
      // debug
      // break;
    }
  }
  fin.close();

#if CC_ALG == MICA
  // mica_db->deactivate(static_cast<uint64_t>(::mica::util::lcore.lcore_id()));
  for (uint64_t thread_id = 0; thread_id < g_thread_cnt; thread_id++) {
    threads.emplace_back([&, thread_id] {
      ::mica::util::lcore.pin_thread(thread_id);
      mem_allocator.register_thread(thread_id);
      mica_db->deactivate(thread_id);
    });
  }
  while (threads.size() > 0) {
    threads.back().join();
    threads.pop_back();
  }
#endif
  return RCOK;
}

template <class IndexT>
void workload::index_insert(IndexT* index, uint64_t key, row_t* row,
                            int part_id) {
#if CC_ALG == MICA
  row = (row_t*)row->get_row_id();
#endif

#if INDEX_STRUCT == IDX_MICA
  auto thread_id = ::mica::util::lcore.lcore_id()>>1; // actual thread
  // printf("thread_id = %lu\n", thread_id);

  MICATransaction tx(mica_db->context(thread_id));
  while (true) {
    // printf("idx=%p part_id=%d key=%lu thread_id=%lu\n", this, part_id, key,
    // thread_id);

    if (!tx.begin()) assert(false);

    auto rc = index->index_insert(NULL, &tx, key, row, part_id);

    if (rc != RCOK) {
      if (!tx.abort()) assert(false);
      continue;
    }
    if (!tx.commit()) continue;

    break;
  }
#else
  auto rc = index->index_insert(NULL, key, row, part_id);
  assert(rc == RCOK);
#endif
}

template <>
void workload::index_insert(IndexArray* index, uint64_t key, row_t* row,
                            int part_id) {
#if CC_ALG == MICA
  row = (row_t*)row->get_row_id();
#endif

  auto rc = index->index_insert(NULL, key, row, part_id);
  assert(rc == RCOK);
}

template <>
void workload::index_insert(IndexMBTree* index, uint64_t key, row_t* row,
                            int part_id) {
#if CC_ALG == MICA
  row = (row_t*)row->get_row_id();
#endif

  auto rc = index->index_insert(NULL, key, row, part_id);
  assert(rc == RCOK);
}

template void workload::index_insert(HASH_INDEX* index, uint64_t key, row_t* row,
                                     int part_id);
template void workload::index_insert(ARRAY_INDEX* index, uint64_t key,
                                     row_t* row, int part_id);
template void workload::index_insert(ORDERED_INDEX* index, uint64_t key,
                                     row_t* row, int part_id);
