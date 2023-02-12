#pragma once
#include "global.h"
#include "table.h"

#include <thread>

// this is the base class for all workload
class workload
{
public:
  // tables indexed by table name
  map<string, table_t*> tables;
  map<string, HASH_INDEX*> hash_indexes;
  map<string, ORDERED_INDEX*> ordered_indexes;

  ::mica::util::Stopwatch mica_sw;

  MICAAlloc* mica_alloc;
  MICAPmemAlloc* mica_aep_alloc;
  MICAPagePool* mica_page_pools[2];
  MICALoggerInterface* mica_logger;
  MICADB*  mica_db;

  void init_mica(uint64_t g_thread_cnt);
  virtual RC init();
  virtual RC init_schema(string schema_file);
  virtual RC init_table();
};

//============================================
//IMPLEMENT

void workload::init_mica (uint64_t g_thread_cnt) {
  printf("MICA initializing\n");

  mica_sw.init_start();
  mica_sw.init_end();

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
}

RC workload::init () {
  return RCOK;
}

RC workload::init_schema (string schema_file) {
  printf ("SCHEMA-TPCC inializing\n");

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
      schema = new Catalog();
      getline(fin, line);
      int col_count = 0;
      // Read all fields for this table.
      vector<string> lines;
      while (line.length() > 1) {
        lines.push_back(line);
        getline(fin, line);
      }
      schema->init(tname.c_str(), lines.size());
      for (uint32_t i = 0; i < lines.size(); i++) {
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
        assert(elem_num == 3 || elem_num == 4);
        schema->add_col((char*)name.c_str(), size, (char*)type.c_str(), cf);
        col_count++;
      }

      int part_cnt = 1;
      if (tname == "ITEM") part_cnt = 1;

      table_t* cur_tab = new table_t();
      new (cur_tab) table_t;

      cur_tab->mica_db = mica_db;

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

      int part_cnt = 1;

      if (tname == "ITEM") part_cnt = 1;

      uint64_t table_size;

      if (tname == "ITEM")
        table_size = stoi(items[1]);
      else
        table_size = stoi(items[1]) * g_num_wh;

      printf (" tname=%s\n", tname.c_str());

      if (strncmp(iname.c_str(), "ORDERED_", 8) == 0) {
        bool ret = mica_db->create_btree_index_unique_u64 (iname, tables[tname]->mica_tbl);
        assert (ret);

        ORDERED_INDEX* index =
        mica_db->get_btree_index_unique_u64 (iname);

        MICATransaction tx(mica_db->context(0));
        index->init(&tx);
        ordered_indexes[iname] = index;

      } else if (strncmp(iname.c_str(), "HASH_", 5) == 0) {
        bool ret = mica_db->create_hash_index_nonunique_u64 (iname, tables[tname]->mica_tbl, table_size*2);
        assert (ret);

        HASH_INDEX* index = 
        mica_db->get_hash_index_nonunique_u64 (iname);

        MICATransaction tx(mica_db->context(0));
        index->init(&tx);
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

  vector<thread> threads;
  // mica_db->deactivate(static_cast<uint64_t>(::mica::util::lcore.lcore_id()));
  for (uint64_t thread_id = 0; thread_id < mica_db->thread_count(); thread_id++) {
    threads.emplace_back([&, thread_id] {
      ::mica::util::lcore.pin_thread(thread_id);
      // mem_allocator.register_thread(thread_id);
      mica_db->deactivate(thread_id);
    });
  }
  while (threads.size() > 0) {
    threads.back().join();
    threads.pop_back();
  }

  printf ("SCHEMA-TPCC inialized\n");
  return RCOK;
}

RC workload::init_table () {
  return RCOK;
}
