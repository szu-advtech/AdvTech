#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#define MAX_COLUMN 1024
#include "SubArray.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"

class Controller
{
public:
	Controller();
	virtual ~Controller();
	/*Functions*/
	void Start_cmd(int bank_num);    //start_cmd to bank_num
	void Finish_cmd(int graph_num, int master_num); //finish_cmd to master_num
	void wait();     //wait for cmd
	void bitmap_init(int size, unsigned  char **bitmap);
	void fetch_parital_bitmap(unsigned char **frontier_bitmap, unsigned char **shared_mm_bitmap); // In this first version, the whole status bimap is copied, later for parital bitmap
	void write_back_to_shared_memory(unsigned char **graph_bitmap, unsigned char **shared_mm_bitmap);  //write back bitmap to shared memory 
	unsigned char * XOR_operation(unsigned char **bitmap_1, unsigned char **bitmap_2);
	unsigned char * AND_operation(unsigned char **bitmap_1, unsigned char **bitmap_2);
	unsigned char *OR_operation(unsigned char **bitmap_1, unsigned char **bitmap_2);
	unsigned char *REVERSE_operation(unsigned char **bitmap);
	void bitmap_set(int index, unsigned char **bitmap);
	int bitmap_get(int index, unsigned char **bitmap);

	/*corssbar operations*/
	int write_cell_subarray(SubArray *subarray, int row_num, int column_num, int value);   // return 0 means successful, while 1 means failed( WDD+SA)
	int read_cell_subarray(SubArray *subarray, int row_num, int column_num);    		// return cell value  (WDD+SA)
    void read_row_subarray(SubArray *subarray, int row_num, int column_num, int row_data[]);     // return a row number of value, (WDD+SA with conflict)

	/*Properties*/
	int register_[32];
	int eDram_0;
	int eDram_1;
	int eDram_2;
	int eDram_3;
	int eDram_4;
	int eDram_5;
	int eDram_6;
	int eDram_7;
	int start_cmd;
	unsigned char * g_bitmap_empty;
	unsigned char * g_bitmap_frontier;
	unsigned char * g_bitmap_status;
	unsigned char * g_bitmap_parent;
	int g_size;  //it should be the vertex of graph divides 8 and plus 1
	int bitmap_number;

};

#endif
