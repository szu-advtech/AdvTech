#include "Trace_collect.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

TraceCollect::TraceCollect()
{
	/*master bank*/
	read_row_for_metadata_in_master=0;
	send_start_cmd=0;
	fetch_status_times_in_master=0;
	writeback_frontier=0;
	save_output=0;

	/*graph bank*/
	fetch_frontier_times=0;
	read_row_for_location=0;
	read_row_for_pre_node_location=0;
	read_row_for_start=0;
	read_row_for_end=0;
	read_row_for_single=0;
	read_row_for_all=0;
	read_eDram_for_check_frontier=0;
	select_adjacent_node_and_update=0;

	fetch_status_times=0;
	writeback_status=0;
	send_finish_cmd=0;
}
TraceCollect::~TraceCollect()
{
	;
}

void TraceCollect::Print_trace_result()
{
	;
}
