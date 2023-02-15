#pragma once
#ifndef TRANSACTION_HASH_TABLE_H_
#define TRANSACTION_HASH_TABLE_H_

#include <stdint.h>
#include <numa.h>

namespace mica {
namespace transaction {

struct Elem {
  uint64_t ts;
  uint64_t row;
  uint16_t cf;
  uint8_t  intention;
  size_t   sz;
  char    *data;
};
struct HashElem {
  uint64_t    pkey;
  struct Elem elem;
};
struct HashCell {
  int      hc_num;
  uint8_t  hc_cas;
  union {
    HashElem hc_elem;
    struct {
      int capacity;
      HashElem *elems;
    } hc_elems;
  } hc_union;

  HashCell () {
    hc_num = 0;
    hc_cas = 0;
  }
};
struct HashTable {
  uint64_t  ht_cell_num;
  uint64_t  ht_elem_num;

  HashCell *ht_cell;
  HashElem *ht_free[16];
  HashElem *ht_elem;
  uint64_t  ht_used_elem_num;

  void init (uint64_t cell_num, uint64_t elem_num) {
    ht_cell_num = cell_num;
    ht_elem_num = elem_num;

    ht_cell = (HashCell*) numa_alloc_onnode (ht_cell_num*sizeof(HashCell), 0);
    ht_elem = (HashElem*) numa_alloc_onnode (ht_elem_num*sizeof(HashElem), 0);

    ht_used_elem_num = 0;
    for (auto i=0; i<16; i++) {
      ht_free[i] = nullptr;
    }
  }
  uint64_t hash (uint64_t pkey) {
    return pkey % ht_cell_num;
  }
  int slot (int capacity) {
    return capacity>>2;
  }
  void destroy () {
    numa_free (ht_cell, ht_cell_num*sizeof(HashCell));
    numa_free (ht_elem, ht_elem_num*sizeof(HashElem));
  }
  void add  (uint64_t pkey, Elem elem) {
    uint64_t cell_id = hash (pkey);
    HashCell &cell = ht_cell[cell_id];
    
    while (__sync_bool_compare_and_swap(&cell.hc_cas, 0, 1) == false)
      ::mica::util::pause ();

    // do useful work
    if (cell.hc_num == 0) {
      cell.hc_union.hc_elem = {pkey, elem};
      cell.hc_num = 1;
    }
    else if (cell.hc_num == 1){
      if (ht_free[0] == nullptr) {
        HashElem *p = ht_elem+ht_used_elem_num;
        ht_used_elem_num += 4;
        p[0] = cell.hc_union.hc_elem;

        cell.hc_union.hc_elems.capacity = 4;
        cell.hc_union.hc_elems.elems = p;
      }
      else {
        HashElem *p = ht_free[0];
        ht_free[0] = *(HashElem**)p;
        p[0] = cell.hc_union.hc_elem;

        cell.hc_union.hc_elems.capacity = 4;
        cell.hc_union.hc_elems.elems = p;
      }
      cell.hc_union.hc_elems.elems[cell.hc_num++] = {pkey, elem};
    }
    else {
      if (cell.hc_num < cell.hc_union.hc_elems.capacity) {
        cell.hc_union.hc_elems.elems[cell.hc_num++] = {pkey, elem};
      }
      else {
        int capacity = cell.hc_union.hc_elems.capacity;
        int slot_id  = slot (capacity);
        if (ht_free[slot_id+1] == nullptr) {
          HashElem *p = ht_elem+ht_used_elem_num;
          ht_used_elem_num += static_cast<uint64_t>(capacity<<1);
          assert (ht_used_elem_num <= ht_elem_num);

          for (auto i=0; i< capacity; i++) {
            p[i] = cell.hc_union.hc_elems.elems[i];
          }
          *(HashElem**)cell.hc_union.hc_elems.elems = ht_free[slot_id];
          ht_free[slot_id] = cell.hc_union.hc_elems.elems;
          cell.hc_union.hc_elems.elems= p;
        }
        else {
          HashElem *p = ht_free[slot_id+1];
          ht_free[slot_id+1] = *(HashElem**)p;
          for (auto i=0; i< capacity; i++) {
            p[i] = cell.hc_union.hc_elems.elems[i];
          }
          *(HashElem**)cell.hc_union.hc_elems.elems = ht_free[slot_id];
          ht_free[slot_id] = cell.hc_union.hc_elems.elems;
          cell.hc_union.hc_elems.elems= p;
        }
        cell.hc_union.hc_elems.elems[cell.hc_num++] = {pkey, elem}; 
      }
    }
    cell.hc_cas = 0;
    ::mica::util::memory_barrier();
  }
};

} // end of namespace transaction
} // end of namespace mica
#endif

