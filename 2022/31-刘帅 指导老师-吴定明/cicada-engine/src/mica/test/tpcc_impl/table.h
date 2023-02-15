#pragma once

#include "catalog.h"
#include "global.h"
#include "txn.h"

// TODO sequential scan is not supported yet.
// only index access is supported for table.
class Catalog;
class row_t;
class table_t;

#define DECL_SET_VALUE(type) \
	void set_value(int col_id, type value);

#define SET_VALUE(type) \
	void row_t::set_value(int col_id, type value) { \
		set_value(col_id, &value); \
	}

#define DECL_GET_VALUE(type)\
	void get_value(int col_id, type & value);

#define GET_VALUE(type)\
	void row_t::get_value(int col_id, type & value) {\
		int pos = get_schema()->get_field_index(col_id);\
		memcpy(&value, data + pos, sizeof(type));\
	}

class row_t
{
public:

	RC init(table_t * host_table, uint64_t part_id, uint64_t row_id = 0);
	void init(int size);
	RC switch_schema(table_t * host_table);

	table_t * get_table();
	Catalog * get_schema();
	const char * get_table_name();
	uint64_t get_field_cnt();
	uint64_t get_tuple_size();
	uint64_t get_row_id() { return _row_id; };

	void copy(row_t * src);

	void 		set_primary_key(uint64_t key) { _primary_key = key; };
	uint64_t 	get_primary_key() {return _primary_key; };
	uint64_t 	get_part_id() { return _part_id; };

	void set_value(int id, void * ptr);
	void set_value(int id, void * ptr, int size);
	void set_value(const char * col_name, void * ptr);
	char * get_value(int id);
	char * get_value(char * col_name);

	DECL_SET_VALUE(uint64_t);
	DECL_SET_VALUE(int64_t);
	DECL_SET_VALUE(double);
	DECL_SET_VALUE(uint32_t);
	DECL_SET_VALUE(int32_t);
	DECL_SET_VALUE(uint16_t);
	DECL_SET_VALUE(int16_t);
	DECL_SET_VALUE(uint8_t);
	DECL_SET_VALUE(int8_t);

	DECL_GET_VALUE(uint64_t);
	DECL_GET_VALUE(int64_t);
	DECL_GET_VALUE(double);
	DECL_GET_VALUE(uint32_t);
	DECL_GET_VALUE(int32_t);
	DECL_GET_VALUE(uint16_t);
	DECL_GET_VALUE(int16_t);
	DECL_GET_VALUE(uint8_t);
	DECL_GET_VALUE(int8_t);


	void set_data(char * data, uint64_t size);
	char * get_data();

public:

	char * data;
};


class table_t
{
public:
	void init(Catalog * schema, uint64_t part_cnt);
	// row lookup should be done with index. But index does not have
	// records for new rows. get_new_row returns the pointer to a
	// new row.
	RC get_new_row(row_t *& row); // this is equivalent to insert()
	RC get_new_row(row_t *& row, uint64_t part_id, uint64_t &row_id);

	void delete_row(); // TODO delete_row is not supportet yet

	// uint64_t get_table_size() { return cur_tab_size; };
	Catalog * get_schema() { return schema; };
	const char * get_table_name() { return table_name; };

	Catalog *  schema;
	MICADB*    mica_db;
	MICATable* mica_tbl;

private:
	const char * 	table_name;
	// uint64_t  		cur_tab_size;
	uint64_t part_cnt;
	char 			pad[CL_SIZE - sizeof(void *)*3];
};

// IMPLEMENT

size_t row_t::alloc_size(table_t* t) { return sizeof(row_t); }
size_t row_t::max_alloc_size() { return sizeof(row_t); }

RC
row_t::init(table_t * host_table, uint64_t part_id, uint64_t row_id) {
	_row_id = row_id;
	_part_id = part_id;
	this->table = host_table;
	is_deleted = 0;
  // debug
  // printf ("row_t::init\n");

  // We ignore the given row_id argument to init() because it contains an
  // uninitialized value and is not used by the workload.

  auto db = table->mica_db;
  auto tbl = table->mica_tbl;

  auto thread_id = ::mica::util::lcore.lcore_id()>>1; // get thread_id from cpu

  // debug
  // printf("lcore_id=%lu thread_id = %lu thread is_active=%d\n",
  //         ::mica::util::lcore.lcore_id(), thread_id, db->is_active(thread_id));

#if defined (test) || defined (CWBL)
  MICATransaction tx(db->context(thread_id));
#elif defined (CZEN)
  MICAPmemTransaction tx(db->context(thread_id));
#endif
        // db->activate(static_cast<uint16_t>(thread_id));
        // tx.aep_begin ();

	Catalog * schema = host_table->get_schema();
	//int tuple_size = schema->get_tuple_size();
	while (true) {

		if (!tx.begin())
			assert(false);

#if defined (test) || defined (CWBL)
		MICAPmemRowAccessHandle rah(&tx);
		if (!rah.new_row(tbl, 0, MICATransaction::kNewRowID, false, schema->cf_sizes[0])) {
			if (!tx.abort())
				assert(false);
			continue;
		}
		_row_id = rah.row_id();
                data = rah.data();
	  if (!tx.commit())
			continue;

		break;
#elif defined (CZEN)
		MICAPmemRowAccessHandle rah(&tx);
		if (!rah.aep_new_row(tbl, 0, MICATransaction::kNewRowID, row_id, false, schema->cf_sizes[0])) {
                        // debug
                        // printf ("storage/row.cpp error aep_new_row!\n");

			if (!tx.aep_abort())
				assert(false);
			continue;
		}
		_row_id = rah.row_id();
                data = rah.data();
	  if (!tx.aep_commit()) {
                        // debug
                        // printf ("storage/row.cpp aep_commit error\n");
			continue;
          }

		break;
#endif
	}
	return RCOK;
}
void
row_t::init(int size)
{
	assert(false);
        is_deleted = 0;
}

RC
row_t::switch_schema(table_t * host_table) {
	assert(false);
	return RCOK;
}

table_t * row_t::get_table() {
	return table;
}

Catalog * row_t::get_schema() {
	return get_table()->get_schema();
}

const char * row_t::get_table_name() {
	return get_table()->get_table_name();
};
uint64_t row_t::get_tuple_size() {
	return get_schema()->get_tuple_size();
}

uint64_t row_t::get_field_cnt() {
	return get_schema()->field_cnt;
}

void row_t::set_value(int id, void * ptr) {
	int datasize = get_schema()->get_field_size(id);
	int pos = get_schema()->get_field_index(id);
	memcpy( &data[pos], ptr, datasize);
}

void row_t::set_value(int id, void * ptr, int size) {
	int pos = get_schema()->get_field_index(id);
	memcpy( &data[pos], ptr, size);
}

void row_t::set_value(const char * col_name, void * ptr) {
	uint64_t id = get_schema()->get_field_id(col_name);
	set_value(id, ptr);
}

SET_VALUE(uint64_t);
SET_VALUE(int64_t);
SET_VALUE(double);
SET_VALUE(uint32_t);
SET_VALUE(int32_t);
SET_VALUE(uint16_t);
SET_VALUE(int16_t);
SET_VALUE(uint8_t);
SET_VALUE(int8_t);

GET_VALUE(uint64_t);
GET_VALUE(int64_t);
GET_VALUE(double);
GET_VALUE(uint32_t);
GET_VALUE(int32_t);
GET_VALUE(uint16_t);
GET_VALUE(int16_t);
GET_VALUE(uint8_t);
GET_VALUE(int8_t);

char * row_t::get_value(int id) {
	int pos = get_schema()->get_field_index(id);
	return &data[pos];
}

char * row_t::get_value(char * col_name) {
	uint64_t pos = get_schema()->get_field_index(col_name);
	return &data[pos];
}

char * row_t::get_data() { return data; }

void row_t::set_data(char * data, uint64_t size) {
	printf("oops\n");
	assert(false);
	memcpy(this->data, data, size);
}

// copy from the src to this
void row_t::copy(row_t * src) {
	printf("oops\n");
	assert(false);
}

void row_t::free_row() {
}

RC row_t::get_row(access_t type, txn_man * txn, row_t *& row) {
	printf("deprecated\n");
	assert(false);
	return ERROR;
}

RC row_t::get_row(access_t type, txn_man* txn, table_t* table,
                  row_t* access_row, uint64_t row_id, uint64_t part_id) {
  assert(part_id >= 0 && part_id < table->mica_tbl.size());

  // printf("get_row row_id=%lu row_count=%lu\n", item->row_id,
  //        table->mica_tbl[this->get_part_id()]->row_count());

    access_row->table = table;
    access_row->set_part_id(part_id);
    access_row->set_row_id(row_id);

    access_t this_type = type;
    uint64_t cf_id = 0;

    if (this_type == PEEK) {

#if defined (test) || defined (CWBL)
      MICARowAccessHandlePeekOnly rah(txn->mica_tx);
      if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, false,
                        false))
#elif defined (CZEN)
      MICAPmemRowAccessHandlePeekOnly rah(txn->mica_tx);
      if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false, false,
                        false))
#endif
        return Abort;

      access_row->data = const_cast<char*>(rah.cdata());
    } else {

#if defined (test) || defined (CWBL)
      MICARowAccessHandle rah(txn->mica_tx);
      if (this_type == RD) {
        if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, true,
                          false) ||
            !rah.read_row())
          return Abort;
      } else if (this_type == WR) {
        if (!rah.peek_row(table->mica_tbl[part_id], cf_id, row_id, false, true,
                          true) ||
            !rah.read_row() || !rah.write_row())
          return Abort;
      } else {
        assert(false);
        return Abort;
      }
#elif defined (CZEN)
      MICAPmemRowAccessHandle rah(txn->mica_tx);
      if (this_type == RD) {
        if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false/*false default*/, true,
                          false) ||
            !rah.aep_read_row()) {

          // debug
          // printf ("storage/row.cpp-1: aep_peek_row/aep_read_row error! row_id=%p\n", (void*)row_id);
          return Abort;
        }
      } else if (this_type == WR) {
        if (!rah.aep_peek_row(table->mica_tbl[part_id], cf_id, row_id, false/*false default*/, true,
                          true) ||
            !rah.aep_read_row() || !rah.aep_write_row(static_cast<uint64_t>(-1))) {

          // debug
          // printf ("storage/row.cpp-2: aep_peek_row/aep_read_row error! row_id=%p\n", (void*)row_id);
          return Abort;
        }
      } else {
        assert(false);
        return Abort;
      }
#endif

      if (this_type == RD)
        access_row->data = const_cast<char*>(rah.cdata());
      else
        access_row->data = rah.data();
    }
  return RCOK;
}

// the "row" is the row read out in get_row().
// For locking based CC_ALG, the "row" is the same as "this".
// For timestamp based CC_ALG, the "row" != "this", and the "row" must be freed.
// For MVCC, the row will simply serve as a version. The version will be
// delete during history cleanup.
// For TIMESTAMP, the row will be explicity deleted at the end of access().
// (cf. row_ts.cpp)

void row_t::return_row(access_t type, txn_man * txn, row_t * row) {
	return;
}


void table_t::init(Catalog* schema, uint64_t part_cnt) {
  this->table_name = schema->table_name;
  this->schema = schema;
  this->part_cnt = part_cnt;

  printf ("table_t:: init part_cnt=%lu \n", part_cnt);

  for (uint64_t part_id = 0; part_id < part_cnt; part_id++) {
    uint64_t thread_id = part_id % mica_db->thread_count();
    ::mica::util::lcore.pin_thread(thread_id);

    static const uint64_t noninlined_data_sizes[] = {4096, 4096, 4096, 4096};
    uint64_t data_sizes[4];
    uint64_t cf_count = schema->cf_count;
 
    for (uint64_t i = 0; i < cf_count; i++)
      data_sizes[i] = schema->cf_sizes[i];

    char buf[1024];
    int i = 0;
    while (true) {
      sprintf(buf, "%s_%d", table_name, i);

      uint64_t table_size = 0;
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
        table_size = mica_db->thread_count()*1024;
      }
      else if (strcmp(table_name, "NEW-ORDER") == 0) {
        table_size = mica_db->thread_count()*1024;
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
      if (mica_db->create_table(buf, cf_count, data_sizes, table_size)) {
        // debug
        printf ("table_t:init create_table table_name=%s table_size=%u\n", table_name, table_size);
        break;
      }
      i++;
    }
    auto p = mica_db->get_table(buf);
    assert(p);

    printf("tbl_name=%s part_id=%" PRIu64 " part_cnt=%" PRIu64 "\n", buf,
           part_id, part_cnt);
    printf("\n");

    mica_tbl = p;
  }
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

  assert(row != NULL);
  return rc;
}

