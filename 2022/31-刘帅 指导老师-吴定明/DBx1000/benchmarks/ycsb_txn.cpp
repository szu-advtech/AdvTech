#include "global.h"
#include "helper.h"
#include "ycsb.h"
#include "ycsb_query.h"
#include "wl.h"
#include "thread.h"
#include "table.h"
#include "row.h"
#include "index_hash.h"
#include "index_btree.h"
#include "catalog.h"
#include "manager.h"
#include "row_lock.h"
#include "row_ts.h"
#include "row_mvcc.h"
#include "mem_alloc.h"
#include "query.h"

void ycsb_txn_man::init(thread_t* h_thd, workload* h_wl, uint64_t thd_id) {
  txn_man::init(h_thd, h_wl, thd_id);
  _wl = (ycsb_wl*)h_wl;
}

RC ycsb_txn_man::run_txn(base_query* query) {

  // debug
  // static int i=0;
  // printf ("ycsb_txn_man::run_txn %d\n", i++);

  RC rc;
  ycsb_query* m_query = (ycsb_query*)query;
  ycsb_wl* wl = (ycsb_wl*)h_wl;
  row_cnt = 0;

  const uint64_t kColumnSize = MAX_TUPLE_SIZE / FIELD_PER_TUPLE;

  uint64_t v = 0;

#if CC_ALG == MICA

#if defined (test) || defined (CWBL)
  mica_tx->begin(false);
#elif defined (CZEN)
  mica_tx->aep_begin();
#endif

#endif

  for (uint32_t rid = 0; rid < m_query->request_cnt; rid++) {
    ycsb_request* req = &m_query->requests[rid];
    int part_id = wl->key_to_part(req->key);
    uint64_t column = req->column;
    bool finish_req = false;
    UInt32 iteration = 0;
    while (!finish_req) {
      access_t type = req->rtype;
      row_t* row = search(_wl->the_index, req->key, part_id, type);

      if (row == NULL) {
        rc = Abort;
        goto final;
      }

      // Computation //
      // Only do computation when there are more than 1 requests.
      /*if (m_query->request_cnt > 1)*/ {
        if (req->rtype == RD || req->rtype == SCAN) {
          //                  for (int fid = 0; fid < schema->get_field_cnt(); fid++) {
#if !TPCC_CF
          const char* data = row->get_data() + column * kColumnSize;
#else
          const char* data = row->cf_data[0] + column * kColumnSize;
#endif
          for (uint64_t j = 0; j < kColumnSize; j += 64)
            v += static_cast<uint64_t>(data[j]);
          v += static_cast<uint64_t>(data[kColumnSize - 1]);
          //                  }
        } else {
          assert(req->rtype == WR);
          //					for (int fid = 0; fid < schema->get_field_cnt(); fid++) {
          //int fid = 0;
          // char * data = row->get_data();
#if !TPCC_CF
          char* data = row->get_data() + column * kColumnSize;
#else
          char* data = row->cf_data[0] + column * kColumnSize;
#endif
          for (uint64_t j = 0; j < kColumnSize; j += 64) {
            v += static_cast<uint64_t>(data[j]);
            data[j] = static_cast<char>(v);
          }
          v += static_cast<uint64_t>(data[kColumnSize - 1]);
          data[kColumnSize - 1] = static_cast<char>(v);
          //*(uint64_t *)(&data[fid * 10]) = 0;
          // memcpy(data, v, column_size);
          //					}
        }
      }

      iteration++;
      if (req->rtype == RD || req->rtype == WR || iteration == req->scan_len)
        finish_req = true;
    }
  }
  rc = RCOK;
final:
  rc = finish(rc);
  return rc;
}
