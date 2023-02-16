#pragma once

#include "global.h"

class row_t;
class table_t;
class IndexHash;
class IndexArray;
class index_btree;
class IndexMBTree;
class IndexMICA;
class OrderedIndexMICA;
class IndexMICAMBTree;
class Catalog;
class lock_man;
class txn_man;
class thread_t;
class index_base;
class Timestamp;
class Mvcc;

// this is the base class for all workload
class workload
{
public:
	// tables indexed by table name
	map<string, table_t *> tables;
	map<string, HASH_INDEX *> hash_indexes;
	map<string, ARRAY_INDEX *> array_indexes;
	map<string, ORDERED_INDEX *> ordered_indexes;

	::mica::util::Stopwatch mica_sw;

#if CC_ALG == MICA
	MICAAlloc* mica_alloc;
        MICAPmemAlloc* mica_aep_alloc;
	MICAPagePool* mica_page_pools[2];
	MICALoggerInterface* mica_logger;
	MICADB* mica_db;
#endif

	void init_mica();

	// initialize the tables and indexes.
	virtual RC init();
	virtual RC init_schema(string schema_file);
	virtual RC init_table()=0;
	virtual RC get_txn_man(txn_man *& txn_manager, thread_t * h_thd)=0;

	bool sim_done;
protected:
	template <class IndexT>
	void index_insert(IndexT* index, uint64_t key, row_t* row, int part_id);
};
