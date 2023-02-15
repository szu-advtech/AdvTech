#pragma once 

#include <map>
#include <vector>
#include <string.h>
#include "helper.h"

class Column {
public:
	Column() {
		this->type = new char[80];
		this->name = new char[80];
	}
	Column(uint64_t size, char * type, char * name, 
		uint64_t id, uint64_t index) 
	{
		this->size = size;
		this->id = id;
		this->index = index;
		this->type = new char[80];
		this->name = new char[80];
		strcpy(this->type, type);
		strcpy(this->name, name);
	};

	uint64_t id;
	uint32_t size;
	uint32_t index;
#if TPCC_CF
        uint64_t cf_id;
#endif
	char * type;
	char * name;
	// char pad[CL_SIZE - sizeof(uint64_t)*3 - sizeof(char *)*2];
};

class Catalog {
public:
	// abandoned init function
	// field_size is the size of each each field.
	void init(const char * table_name, int field_cnt) {
	  this->table_name = strdup(table_name);
	  this->field_cnt = 0;
	  this->_columns = new Column [field_cnt];
	  this->tuple_size = 0;
          cf_count = 0;
          memset(cf_sizes, 0, sizeof(cf_sizes));
	}

	void add_col(char * col_name, uint64_t size, char * type, int cf_id) {
          #if !TPCC_CF
            assert(cf_id == 0);
          #endif
          
            assert((size_t)cf_id < sizeof(cf_sizes) / sizeof(cf_sizes[0]));
            if (cf_count < (uint64_t)cf_id + 1)
              cf_count = (uint64_t)cf_id + 1;
          
          	_columns[field_cnt].size = size;
          	strcpy(_columns[field_cnt].type, type);
          	strcpy(_columns[field_cnt].name, col_name);
          	_columns[field_cnt].id = field_cnt;
          	_columns[field_cnt].index = cf_sizes[cf_id];
          #if TPCC_CF
          	_columns[field_cnt].cf_id = cf_id;
          #endif
                  cf_sizes[cf_id] += size;
          	tuple_size += size;
          	field_cnt ++;
	}

	uint32_t 			field_cnt;
 	const char * 	table_name;
	
	uint32_t		get_tuple_size() { return tuple_size; };
	uint64_t 		get_field_cnt() { return field_cnt; };
	uint64_t 		get_field_size(int id) { return _columns[id].size; };
	uint64_t 		get_field_index(int id) { return _columns[id].index; };
#if TPCC_CF
	uint64_t 		get_field_cf_id(int id) { return _columns[id].cf_id; };
#endif
	char * 			get_field_type(uint64_t id) { return _columns[id].type; }
	char * 			get_field_name(uint64_t id) { return _columns[id].name;  }
	uint64_t 		get_field_id(const char * name) {
				uint32_t i;
				for (i = 0; i < field_cnt; i++) {
					if (strcmp(name, _columns[i].name) == 0)
						break;
				}
				assert (i < field_cnt);
				return i;
	}
	char * 			get_field_type(char * name) {  return get_field_type( get_field_id(name) );}
	uint64_t 		get_field_index(char * name) { return get_field_index( get_field_id(name) );  }

	void 			print_schema() {
				printf("\n[Catalog] %s\n", table_name);
				for (uint32_t i = 0; i < field_cnt; i++) {
					printf("\t%s\t%s\t%ld\n", get_field_name(i),
						get_field_type(i), get_field_size(i));
				}
	}

	Column * 		_columns;
	uint32_t 		tuple_size;

        uint64_t                cf_count;
        uint64_t                cf_sizes[4];
};

