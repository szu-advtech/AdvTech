#ifndef TRACECOLLECT_H_
#define TRACECOLLECT_H_

#include <iostream>
#include <string>
#include <stdint.h>

using namespace std;

class TraceCollect {

public:
	TraceCollect();
	virtual ~TraceCollect();

	/*Function*/
	void Print_trace_result();

	/*Property*/
	//master bank operations
	int read_row_for_metadata_in_master;
	int send_start_cmd;
	int fetch_status_times_in_master;
	int writeback_frontier;
	int save_output;

	//graph banks oeprations
	int fetch_frontier_times;
	int read_row_for_location;
	int read_row_for_pre_node_location; //if node and pre_node are located in same row, then this value shoud be 0
	int read_row_for_start;
	int read_row_for_all;
	int read_row_for_single;
	int read_row_for_end;
	int read_eDram_for_check_frontier;
	int select_adjacent_node_and_update;

    int fetch_status_times;
	int writeback_status;
	int send_finish_cmd;
};
#endif
