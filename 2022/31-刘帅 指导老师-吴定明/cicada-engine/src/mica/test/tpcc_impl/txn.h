#pragma once
#include "global.h"

class workload;
class thread_t;
class base_query;

class txn_man
{
public:
	virtual void init(thread_t * h_thd, workload * h_wl) = 0;
	thread_t * h_thd;
	workload * h_wl;

	virtual 		RC run_txn(base_query * m_query) = 0;
	RC 			finish(RC rc);
	void 			cleanup(RC rc);

#if defined (test) || defined (CWBL)
  	MICATransaction* mica_tx;
#elif defined (CZEN)
  	MICAPmemTransaction* mica_tx;
#endif

};

//================================================================================
//IMPLEMENT

void txn_man::init(thread_t * h_thd, workload * h_wl, uint64_t thd_id) {
	this->h_thd = h_thd;
	this->h_wl = h_wl;
	this->txn_id = txn_id;

#if defined (test) || defined (CWBL)
	mica_tx = new MICATransaction(h_wl->mica_db->context(thd_id));
#elif defined (CZEN)
	mica_tx = new MICAPmemTransaction(h_wl->mica_db->context(thd_id));
#endif
}

// IMPLEMENT

void txn_man::cleanup(RC rc) {
  // run transaction abort
}

RC txn_man::finish(RC rc) {

#if defined (test) || defined (CWBL)
  if (rc == RCOK) {
    if (mica_tx->has_began()) {
      auto write_func = [this]() { return apply_index_changes(RCOK) == RCOK; };
      rc = mica_tx->commit(NULL, write_func) ? RCOK : Abort;
      if (rc != RCOK) rc = apply_index_changes(rc);
    } else
      rc = RCOK;
  } else if (mica_tx->has_began() && !mica_tx->abort())
    assert(false);
    cleanup(rc);
#elif defined (CZEN)
  if (rc == RCOK) {
    if (mica_tx->has_began()) {
      rc = mica_tx->aep_commit() ? RCOK : Abort;
    } else
      rc = RCOK;
  } else if (mica_tx->has_began() && !mica_tx->aep_abort())
    assert(false);
    cleanup(rc);
#endif

  return rc;
}

