#include "Controller.h"
#include <sys/time.h>
#include <time.h>


Controller:: Controller()
{
	int i;
	for(i=0;i<32;i++)
		register_[i]=0;
	eDram_0=0;
	eDram_1=0;
	eDram_2=0;
	eDram_3=0;
	eDram_4=0;
	eDram_5=0;
	eDram_6=0;
	eDram_7=0;
	g_bitmap_empty=NULL;
	g_bitmap_frontier=NULL;
	g_bitmap_status=NULL;
	g_bitmap_parent=NULL;
	g_size=0;  //it should be the vertex of graph divides 8 and plus 1
	bitmap_number=0;
	start_cmd=0;
}
Controller::~Controller ()
{
	delete g_bitmap_empty;
	delete g_bitmap_frontier;
	delete g_bitmap_status;	
	delete g_bitmap_parent;
}
void Controller::Start_cmd(int bank_num)
{
	;	
}
void Controller::Finish_cmd(int graph_num, int master_num)
{
	;
}
void Controller::wait() // here we can record reallife time and use frenquency to evalutate it
{
	;
} 
void Controller::bitmap_init(int size, unsigned  char **bitmap)
{
	bitmap_number=size;
	*bitmap=(unsigned char *)malloc((size/8+1)*sizeof(char));
	if(*bitmap==NULL)
		return ;
	g_size=size/8+1;
	//printf("size is %d g_size is %d\n", size, g_size);
	memset(*bitmap,0x0, g_size);
}

void Controller::fetch_parital_bitmap(unsigned char **frontier_bitmap, unsigned char **shared_mm_bitmap)   // In this first version, the whole status bimap is copied, later for parital bitmap
{
	//memcpy(*frontier_bitmap, *shared_mm_bitmap, g_size);
	int i;
	memset(*frontier_bitmap, 0x0, g_size);
	for(i=0;i<bitmap_number;i++)
	{
		int temp=bitmap_get(i, shared_mm_bitmap);
		if(temp==1)
			bitmap_set(i, frontier_bitmap);
	}
}
void Controller::write_back_to_shared_memory(unsigned char **graph_bitmap, unsigned char **shared_mm_bitmap)  //write back bitmap to shared memory 
{
	//printf("g_size is %d\n", g_size);
	//memcpy(shared_mm_bitmap, graph_bitmap, g_size);
	
	int i;
	memset(*shared_mm_bitmap,0x0, g_size);
	for(i=0;i<bitmap_number;i++)
	{
		int temp=bitmap_get(i, graph_bitmap);
		if(temp==1)
			bitmap_set(i, shared_mm_bitmap);
	}
}
unsigned char * Controller::XOR_operation(unsigned char **bitmap_1, unsigned char **bitmap_2)
{
	int i;
	int temp_1;
	int temp_2;
	unsigned char *result_bitmap=NULL;
	bitmap_init(bitmap_number, &result_bitmap);
	for(i=0;i<bitmap_number;i++)
	{
		temp_1=bitmap_get(i, bitmap_1);
		temp_2=bitmap_get(i, bitmap_2);
		if(temp_1==1&&temp_2==0)
			bitmap_set(i, &result_bitmap);
	}
	
	return result_bitmap;

}
unsigned char * Controller::AND_operation(unsigned char **bitmap_1, unsigned char **bitmap_2)
{
	 int i;
        int temp_1;
        int temp_2;
        //unsigned char *result_bitmap=NULL;
        //bitmap_init(bitmap_number, &result_bitmap);
        for(i=0;i<bitmap_number;i++)
        {   
                temp_1=bitmap_get(i, bitmap_1);
                temp_2=bitmap_get(i, bitmap_2);
                if(1==temp_1&&1==temp_2)
                        bitmap_set(i, bitmap_1);
        }   
        return *bitmap_1;

}
unsigned char *Controller::OR_operation(unsigned char **bitmap_1, unsigned char **bitmap_2)
{
	int i;
        int temp_1;
        int temp_2;
        unsigned char *result_bitmap=NULL;   
        bitmap_init(bitmap_number, &result_bitmap);
        for(i=0;i<bitmap_number;i++)
        {
                temp_1=bitmap_get(i, bitmap_1);
                temp_2=bitmap_get(i, bitmap_2);
                if(1==temp_1||1==temp_2)
                        bitmap_set(i, &result_bitmap);
        }
        return result_bitmap;
}
unsigned char *Controller::REVERSE_operation(unsigned char **bitmap)
{
	int i;
	int temp;
	unsigned char *result_bitmap=NULL;
	bitmap_init(bitmap_number, &result_bitmap);
	for(i=0; i< bitmap_number; i++)
	{
		temp=bitmap_get(i, bitmap);
		if(temp==0)
			bitmap_set(i, &result_bitmap);
		
	}
	return result_bitmap;

}

void Controller::bitmap_set(int index, unsigned char **bitmap)
{
	int quo=(index)/8;
	int remainder=(index)%8;
	unsigned char x=(0x1<<remainder);
	if(quo > g_size)
		return ;
	(*bitmap)[quo] =(*bitmap)[quo] |  x;
}
int Controller::bitmap_get(int index, unsigned char **bitmap)
{
        int quo=(index)/8;
        int remainder=(index)%8;
        unsigned char x=(0x1<<remainder);
	unsigned char res;
        if(quo > g_size)
                return -1;
        res=(*bitmap)[quo] & x;
	return res > 0 ? 1: 0;
}   

/*crossbar operations*/
int Controller::write_cell_subarray(SubArray *subarray, int row_num, int column_num, int value)  //return 0 means successful, while 1 means failed 
{
	return subarray->write_cell(row_num, column_num, value);
	
}
int Controller::read_cell_subarray(SubArray *subarray, int row_num, int column_num) //return cell value (WDD+SA)  return -1 means failed return value means successful
{
        return subarray->read_cell(row_num, column_num);
}
void Controller::read_row_subarray(SubArray *subarray, int row_num, int column_num, int row_data[])  // return a row number of value (WDD+SA) with conflict
{
	int i;
	for(i=0;i<column_num;i++)
	{
		int temp;
		temp=subarray->read_cell(row_num, i);	
		if(temp==-1)
			printf("Read cell in this subarray has failed\n");
		else 
			row_data[i]=temp;
	}

}
