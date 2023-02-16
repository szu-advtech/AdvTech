#pragma once
#include "global.h"

class workload;
class base_query;

class thread_t {
public:
  uint64_t _thd_id;
  workload * _wl;

  void     init(uint64_t thd_id, workload * workload);
  RC       run();

  struct   timeval _tv_start;
  struct   timeval _tv_end;
  
  uint64_t _total_neworder, _commit_neworder;
  uint64_t _total_payment , _commit_payment;
  uint64_t _total_delivery, _commit_delivery;
  uint64_t _total_orderstatus, _commit_orderstatus;
  uint64_t _total_stocklevel , _commit_stocklevel;

  uint64_t get_thd_id () { return _thd_id; }
};

//============================================================
//IMPLEMENT

void thread_t::init(uint64_t thd_id, workload * workload) {
  this->_thd_id = thd_id;
  this->_wl = workload;
}

RC thread_t::run() {
        ::mica::util::lcore.pin_thread(get_thd_id());
	_wl->mica_db->activate(static_cast<uint16_t>(get_thd_id()));
	while (_wl->mica_db->active_thread_count() < g_thread_cnt) {
		PAUSE;
		_wl->mica_db->idle(static_cast<uint16_t>(get_thd_id()));
	}

	RC rc = RCOK;
	txn_man * m_txn;
	rc = _wl->get_txn_man(m_txn, this);

	base_query * m_query = nullptr;
	uint64_t thd_txn_id = 0;

	gettimeofday (&_tv_start, nullptr);
	while (thd_txn_id < g_transaction_cnt) {
		m_query = query_queue->get_next_query(_thd_id);
		m_txn->set_txn_id(get_thd_id() + thd_txn_id * g_thread_cnt);
		thd_txn_id ++;
		rc = m_txn->run_txn(m_query);
	}
	gettimeofday (&_tv_end,   nullptr);
	return RCOK;
}
