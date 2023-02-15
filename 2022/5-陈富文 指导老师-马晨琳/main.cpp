#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include "jerasure.h"
#include "cauchy.h"
#include "PIM_InputParameter.h"
#include "SubArray.h"
#include "Bank.h"
#include "Controller.h"
#include "Trace_collect.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#define PERCENTAGE 0.8
#define CACHE_NUM 6
using namespace std;

InputParameter *PIM_InputParameter;
Bank *bank[512];

TraceCollect *tos[512];

int node_num; //  variables for shared memory

void Init_bank();
void Init_trace();
void Print_bank(Bank *bank);
void Init_with_crs();
void Init_controller(int i);
void Print_trace_result();
void Encoding();
void Flush_to_disk(int merge[1024][64]);
void Remove_disk_file();
void Constr_multi_fail_decode_file();
void Init_crs_in_mul_fail(vector<int> disk_used);
void Decode_multi_fail();
void Flush_to_decode_disk(int merge[1024][64]);
int Constr_Sing_fail_decode_file();
void Decode_Sing_fail();
int FileSize(const char* fname); // 获得文件大小
void Encoding_combination();
void Init_crs_of_combination();
void Flush_to_disk_combination(int merge[1024][64]);
void Constr_multi_fail_decode_file_combination();
void Init_crs_in_mul_fail_combination(vector<int> disk_used);
void Decode_multi_fail_combination();
void Flush_to_decode_disk_combination(int merge[1024][64]);
int Constr_Sing_fail_decode_file_combination();
void Decode_Sing_fail_combination();

int k = 10;
int m = 2;
int w = 8;
int stripe = 512; // input数组的大小
int size_of_input = 0;

int ec_num = 1;

int main()
{
	srand48(0);
	FILE *fd_result;
	fd_result = fopen("result.txt", "a+");
	string inputFileName = "sample.cfg";
	// k = atoi(argv[1]);
	// m = atoi(argv[2]);
	// w = atoi(argv[3]);
	cout << "k = " << k << " m = " << m << " w = " << w << endl;

	Remove_disk_file(); // delete historical input data file
	PIM_InputParameter = new InputParameter();
	// RESTORE_SEARCH_SIZE;
	PIM_InputParameter->ReadInputParameterFromFile(inputFileName);
	// record output
	Init_trace();
	// set tos trace information
	for (int i = 0; i < ec_num; i++)
	{
		tos[i]->read_row_for_all = 0;	// active wordlines
		tos[i]->writeback_status = 0;	// save stripe on reram for flushing to disk
		tos[i]->fetch_status_times = 0; // slice data times
	}
	// init reram bank
	Init_bank();
	/*Start to encoding*/
	Encoding();
	Print_trace_result();
	// Encoding_combination();
	fclose(fd_result);
	cout << "Encoding result : " << endl;
	// Print_trace_result();
	/*Start to decoding*/
	Decode_multi_fail();
	// Decode_multi_fail_combination();
	cout << "Mul Fail Decoding result : " << endl;
	Print_trace_result();
	Decode_Sing_fail();
	// Decode_Sing_fail_combination();
	cout << "Singal Fail Decoding result : " << endl;
	Print_trace_result();
	return 0;
}

void Print_trace_result()
{
	double sum = 0;
	float tRP = 0.5;
	float tRCD = 22.5;
	float tCL = 9.8;
	float tWR = 41.4;
	float bus_bandwidth = 0.1; // the time for checking ont byte
	float SenseAmp = 1.454;
	float Subarray_latency = 10.447;
	float shift_and_add = 5;
	float read_slice_data = 1 * w * k; // read a column data

	int inter_para_num = 4;
	int intra_para_num = 128; // a stripe are processing in 128 ReRAM subarray
	int reram_para_num = 16;  // how many banks for a stripe encoding

	// float read_row_latency=tRP+tRCD+tCL+SenseAmp+shift_and_add;
	float read_row_latency = 48;
	float write_row_latency = 48;
	float mapping_latency = 1;

	// printf("read_row: %d, writeback: %d\n", tos[0]->read_row_for_all, tos[0]->writeback_status);
	sum = read_row_latency * tos[0]->read_row_for_all / (float)inter_para_num / (float)intra_para_num / reram_para_num; // encoding time
	sum = sum + write_row_latency * tos[0]->writeback_status * m / (float)reram_para_num;								// only need to save parity cell

	sum = sum + tos[0]->fetch_status_times * read_slice_data / (float)intra_para_num; // fetch data
	printf("read_row: %d writeback: %d fetch_data:%d\n", tos[0]->read_row_for_all, tos[0]->writeback_status, tos[0]->fetch_status_times);
	printf("The whole latency is %lf(ns)\n", sum);
	printf("The whole performace is %lf(MB/s)\n\n", ((double)size_of_input / 1024 / 1024) / (sum / 1000 / 1000 / 1000));
	for (int i = 0; i < ec_num; i++)
	{
		tos[i]->read_row_for_all = 0;	// active wordlines
		tos[i]->writeback_status = 0;	// save stripe on reram for flushing to disk
		tos[i]->fetch_status_times = 0; // slice data times
	}
}

void Init_bank()
{
	int i;
	for (i = 0; i < PIM_InputParameter->bank; i++)
	{
		bank[i] = new Bank();
		bank[i]->bank_num = i;
		if (i == 0) // init bank_mode bank[0] is master
		{
			bank[i]->mode = 0;
			bank[i]->master_bank_num = -1;
		}
		else
		{
			bank[i]->mode = 1;
			bank[i]->master_bank_num = 0;
		}
		bank[i]->num_subarray = PIM_InputParameter->matrix;
		// init controller for each bank
		bank[i]->controller = new Controller();
		Init_controller(i);
		int j;
		for (j = 0; j < bank[i]->num_subarray; j++)
		{
			bank[i]->subarray[j] = new SubArray(); // bank->SubArray=Matrix
			int rownum = PIM_InputParameter->rowsubarray;
			int columnnum = PIM_InputParameter->columnsubarray;
			bank[i]->subarray[j]->Initialize(rownum, columnnum, 1);
		}
	}
}

void Init_trace()
{
	int i;
	for (i = 0; i < 512; i++)
	{
		tos[i] = new TraceCollect();
	}
}

void Init_controller(int i)
{
	bank[i]->controller->bitmap_init(node_num, &bank[i]->controller->g_bitmap_empty);
	bank[i]->controller->bitmap_init(node_num, &bank[i]->controller->g_bitmap_status);
	bank[i]->controller->bitmap_init(node_num, &bank[i]->controller->g_bitmap_frontier);
	bank[i]->controller->bitmap_init(node_num, &bank[i]->controller->g_bitmap_parent);
}

void Init_with_crs()
{
	int *matrix, *bitmatrix;
	matrix = cauchy_original_coding_matrix(k, m, w);
	// concluent CV matrix
	for (int i = 0; i < k; i++)
	{
		matrix[i] = 1;
	}

	bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);

	// init vandermonde matrix
	int array_num = PIM_InputParameter->matrix;

	int i, j; // temp variable
	for (i = 0; i < w * m; i++)
	{
		for (j = 0; j < w * k; j++)
		{
			bank[0]->subarray[0]->cells[j][i] = bitmatrix[i * (w * k) + j];
		}
	}
	tos[0]->fetch_status_times++;
	// duplicate subarray
	for (int h = 1; h < array_num; h++)
	{
		for (i = 0; i < w * k; i++)
		{
			for (j = 0; j < w * m; j++)
				bank[0]->subarray[h]->cells[i][j] = bank[0]->subarray[0]->cells[i][j];
		}
		tos[0]->fetch_status_times++;
	}
	//printf("Finish mapping matrix\n");
}

void Encoding()
{
	// constrcut a genertor matrix to corresponding bank and surray
	Init_with_crs();
	int fp = open("./Input_data/input_small.file", O_RDONLY);
	size_of_input = FileSize("./Input_data/input_small.file");
	cout << "input size is " << size_of_input << endl;

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{
			int ret = read(fp, buf, 8);
			tos[0]->fetch_status_times++;
			if (ret == 0)
			{
				finish_read = true;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
			// merge output for data
			for (int i = 0; i < 64; i++)
				merge[count][i] = input[count][i];
		}
		// parity computation
		int output[stripe][64];
		for (int h = 0; h < 64; h++)
		{
			for (int i = 0; i < w * m; i++)
			{
				int tmp_sum = 0;
				for (int j = 0; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[0]->cells[j][i] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			tos[0]->read_row_for_all++;
		}
		// merge output for parity
		for (int i = 0; i < w * m; i++)
		{
			for (int j = 0; j < 64; j++)
				merge[i + w * k][j] = output[i][j];
		}
		// flush to disk
		Flush_to_disk(merge);
		tos[0]->writeback_status++;
	}
	//printf("Finish!\n");
}

void Flush_to_disk(int merge[1024][64])
{
	// flush to disk
	for (int i = 0; i < (m + k); i++)
	{
		char diskname[128];
		sprintf(diskname, "./Disk_data/Disk_data_%d", i);
		ofstream file(diskname, fstream::app);
		// content
		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (merge[j + i * w][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
}

void Remove_disk_file()
{
	for (int i = 0; i < (m + k); i++)
	{
		char diskname[128];
		sprintf(diskname, "./Disk_data/Disk_data_%d", i);
		remove(diskname);
		sprintf(diskname, "./Disk_data_combination/Disk_data_combination_%d", i);
		remove(diskname);
	}
	for (int i = 0; i < k; i++)
	{
		char diskname[128];
		sprintf(diskname, "./Decode_disk_data/Decode_disk_data_%d", i);
		remove(diskname);
		sprintf(diskname, "./Decode_disk_data_combination/Decode_disk_data_combination_%d", i);
		remove(diskname);
	}
	remove("./Decode_disk_data_combination/decode_data_small_combination.file");
	remove("./Decode_disk_data_combination/decode_singal_data_small_combination.file");
	remove("./Decode_disk_data/decode_data_small.file");
	remove("./Decode_disk_data/decode_singal_data_small.file");
}

void Constr_multi_fail_decode_file()
{
	// 随机选择m个文件作为故障文件
	vector<int> erased(m);
	vector<int> is_disk_failed(k + m);
	for (int i = 0; i < k + m; ++i)
		is_disk_failed[i] = 0;
	cout << "Failed disks is ";
	for (int i = 0; i < m;)
	{
		erased[i] = lrand48() % (k + m);
		if (is_disk_failed[erased[i]] == 0)
		{
			is_disk_failed[erased[i]] = 1;
			printf("%d ", erased[i]);
			i++;
		}
	}
	cout << endl;

	//cout << " use disks is ";
	vector<int> disk_used(k); // 记录decode用的是哪些块的数据
	for (int i = 0, index = 0; i < k; index++)
	{
		if (is_disk_failed[index] == 1)
		{
			continue;
		}
		disk_used[i] = index;
		//cout << index << " ";
		i++;
	}
	//cout << endl;

	//将未发生故障的数据块和校验块写入同一个文件中作为decode的输入
	ofstream decode_data_file("./Decode_disk_data/decode_data_small.file", fstream::app);
	vector<int> open_disk_file(k);
	char diskname[128];
	char buf[stripe];
	for (int i = 0; i < k; i++)
	{ // 打开要读取的文件
		sprintf(diskname, "./Disk_data/Disk_data_%d", disk_used[i]);
		open_disk_file[i] = open(diskname, O_RDONLY);
	}

	while (1)
	{ // 每次循环依次从disk读取w * 8字节，直到disk读完
		int ret;
		for (int i = 0; i < k; i++)
		{
			for (int num = 0; num < w; num++)
			{ // 每个文件一次读取w行, 一行为8字节
				ret = read(open_disk_file[i], buf, 8);
				for (int j = 0; j < ret; j++)
				{
					decode_data_file << buf[j];
				}
			}
		}
		if (ret == 0)
			break; //因为每个disk中字节数都一样，所以会一同完成读取
	}
	for (int i = 0; i < k; i++)
	{ // 关闭文件
		close(open_disk_file[i]);
	}
	Init_crs_in_mul_fail(disk_used);
}

void Init_crs_in_mul_fail(vector<int> disk_used)
{
	int *matrix, *bitmatrix, *decode_matrix, *inv_decode_matrix;
	matrix = cauchy_original_coding_matrix(k, m, w);
	// concluent CV matrix
	for (int i = 0; i < k; i++)
	{
		matrix[i] = 1;
	}
	decode_matrix = new int[k * k];
	inv_decode_matrix = new int[k * k];
	int i = 0, j = 0;
	for (i = 0; i < k; i++)
	{
		if (disk_used[i] < k)
		{
			for (j = 0; j < k; j++)
			{ // 选择的数据块在单位矩阵中的对应行
				if (j == disk_used[i])
					decode_matrix[i * k + j] = 1;
				else
					decode_matrix[i * k + j] = 0;
			}
		}
		else
		{
			for (j = 0; j < k; j++)
			{ // 选择的校验块对应的编码矩阵行
				decode_matrix[i * k + j] = matrix[(disk_used[i] - k) * k + j];
			}
		}
	}

	int tmp = jerasure_invert_matrix(decode_matrix, inv_decode_matrix, k, w);
	bitmatrix = jerasure_matrix_to_bitmatrix(k, k, w, inv_decode_matrix);

	// 映射矩阵
	int array_num = PIM_InputParameter->matrix;

	int q; // temp variable
	for (i = 0; i < w * k; i++)
	{
		for (j = 0; j < w * k; j++)
		{
			bank[0]->subarray[0]->cells[j][i] = bitmatrix[i * (w * k) + j];
		}
	}
	tos[0]->fetch_status_times++;

	// duplicate subarray
	for (int h = 1; h < array_num; h++)
	{
		for (i = 0; i < w * k; i++)
		{
			for (j = 0; j < w * k; j++)
				bank[0]->subarray[h]->cells[i][j] = bank[0]->subarray[0]->cells[i][j];
		}
		tos[0]->fetch_status_times++;
	}
	//printf("Finish mapping decode matrix\n");
}

void Decode_multi_fail()
{
	Constr_multi_fail_decode_file();
	int fp = open("./Decode_disk_data/decode_data_small.file", O_RDONLY);

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{ // 读取文件并转为二进制
			int ret = read(fp, buf, 8);
			tos[0]->fetch_status_times++;
			if (ret == 0)
			{
				finish_read = true;
				break;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
		}
		if(finish_read) break;
		// decode data
		int output[stripe][64];
		for (int h = 0; h < 64; h++)
		{
			for (int i = 0; i < w * k; i++)
			{
				int tmp_sum = 0;
				for (int j = 0; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[0]->cells[j][i] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			tos[0]->read_row_for_all++;
		}
		// merge output for parity
		for (int i = 0; i < w * k; i++)
		{
			for (int j = 0; j < 64; j++)
				merge[i][j] = output[i][j];
		}
		// flush to disk
		Flush_to_decode_disk(merge);
		tos[0]->writeback_status++;
	}
	//printf("Finish!\n");
}

void Flush_to_decode_disk(int merge[1024][64])
{
	for (int i = 0; i < k; i++)
	{
		char diskname[128];
		sprintf(diskname, "./Decode_disk_data/Decode_disk_data_%d", i);
		ofstream file(diskname, fstream::app);
		// content
		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (merge[j + i * w][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
}

int Constr_Sing_fail_decode_file()
{
	int failed_disk = lrand48() % k;
	cout << "Failed disks is " << failed_disk << endl;
	//cout << " use disks is ";
	vector<int> disk_used(k); // 记录decode用的是哪些块的数据
	for (int i = 0, index = 0; i < k; index++)
	{
		if (index == failed_disk)
		{
			continue;
		}
		disk_used[i] = index;
		//cout << index << " ";
		i++;
	}
	//cout << endl;

	//将未发生故障的数据块和校验块写入同一个文件中作为decode的输入
	ofstream decode_data_file("./Decode_disk_data/decode_singal_data_small.file", fstream::app);
	vector<int> open_disk_file(k);
	char diskname[128];
	char buf[stripe];
	for (int i = 0; i < k; i++)
	{ // 打开要读取的文件
		sprintf(diskname, "./Disk_data/Disk_data_%d", disk_used[i]);
		open_disk_file[i] = open(diskname, O_RDONLY);
	}

	while (1)
	{
		int ret;
		for (int i = 0; i < k; i++)
		{
			for (int num = 0; num < w; num++)
			{ // 每个文件一次读取w行, 一行为8字节
				ret = read(open_disk_file[i], buf, 8);
				for (int j = 0; j < ret; j++)
				{
					decode_data_file << buf[j];
				}
			}
		}
		if (ret == 0)
			break; //当前磁盘完成读取
	}
	for (int i = 0; i < k; i++)
	{ // 关闭文件
		sprintf(diskname, "./Disk_data/Disk_data_%d", disk_used[i]);
		close(open_disk_file[i]);
	}
	return failed_disk;
}

void Decode_Sing_fail()
{
	int failed_disk = Constr_Sing_fail_decode_file();

	int fp = open("./Decode_disk_data/decode_singal_data_small.file", O_RDONLY);
	char diskname[128];
	sprintf(diskname, "./Decode_disk_data/Decode_sing_disk_data_%d", failed_disk);
	remove(diskname);

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{ // 读取文件并转为二进制
			int ret = read(fp, buf, 8);
			if (ret == 0)
			{
				finish_read = true;
				break;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
		}

		if(finish_read) break;
		
		// decode data
		int output[stripe][64];
		for (int h = 0; h < 64; h++)
		{
			for (int i = 0; i < w; i++)
			{
				int tmp_sum = 0;
				for (int j = 0; j < k; j++)
				{
					tmp_sum += input[j * w + i][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			//tos[0]->fetch_status_times++;
			tos[0]->read_row_for_all++;
		}

		// flush to disk
		tos[0]->writeback_status++;

		ofstream file(diskname, fstream::app);

		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (output[j][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
	//printf("Finish!\n");
}

int FileSize(const char* fname)
{
    struct stat statbuf;
    if(stat(fname,&statbuf)==0)
        return statbuf.st_size;
    return -1;
}

void Init_crs_of_combination(){
	int *matrix, *bitmatrix;
	matrix = cauchy_original_coding_matrix(k, m, w);
	// concluent CV matrix
	for (int i = 0; i < k; i++)
	{
		matrix[i] = 1;
	}

	bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);
	jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w);

	// init vandermonde matrix
	int array_num = PIM_InputParameter->matrix;

	int i, j; // temp variable
	int row_mid = w * k / 2;
	int col_mid = m * w / 2;
	for (i = 0; i < col_mid; i++)
	{
		for (j = 0; j < row_mid; j++)
		{
			bank[0]->subarray[0]->cells[j + 1][i] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = col_mid; i < w * m; i++)
	{
		for (j = 0; j < row_mid; j++)
		{
			bank[0]->subarray[1]->cells[j + 1][i - col_mid] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = 0; i < col_mid; i++)
	{
		for (j = row_mid; j < w * k; j++)
		{
			bank[0]->subarray[2]->cells[j - row_mid + 1][i] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = col_mid; i < w * m; i++)
	{
		for (j = row_mid; j < w * k; j++)
		{
			bank[0]->subarray[3]->cells[j - row_mid + 1][i - col_mid] = bitmatrix[i * (w * k) + j];
		}
	}
}

void Encoding_combination(){
	// constrcut a genertor matrix to corresponding bank and surray
	Init_crs_of_combination();
	int fp = open("./Input_data/input_small.file", O_RDONLY);
	size_of_input = FileSize("./Input_data/input_small.file");
	cout << "input size is " << size_of_input << endl;

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{
			int ret = read(fp, buf, 8);
			tos[0]->fetch_status_times++;
			if (ret == 0)
			{
				finish_read = true;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
			// merge output for data
			for (int i = 0; i < 64; i++)
				merge[count][i] = input[count][i];
		}
		// parity computation
		int output[stripe][64];
		int i, j;
		int row_mid = w * k / 2;
		int col_mid = m * w / 2;
		for (int h = 0; h < 64; h++)
		{
			for (i = 0; i < col_mid; i++)
			{
				int tmp_sum = 0;
				for (j = 0; j < row_mid; j++)
				{
					tmp_sum += bank[0]->subarray[0]->cells[j + 1][i] * input[j][h];
				}
				bank[0]->subarray[2]->cells[0][i] = tmp_sum % 2;
			}
			for (i = col_mid; i < w * m; i++)
			{
				int tmp_sum = 0;
				for (j = 0; j < row_mid; j++)
				{
					tmp_sum += bank[0]->subarray[1]->cells[j + 1][i - col_mid] * input[j][h];
				}
				bank[0]->subarray[3]->cells[0][i - col_mid] = tmp_sum % 2;
			}
			for (i = 0; i < col_mid; i++)
			{
				int tmp_sum = 0;
				tmp_sum += bank[0]->subarray[2]->cells[0][i] * 1;
				for (j = row_mid; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[2]->cells[j - row_mid + 1][i] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			for (i = col_mid; i < w * m; i++)
			{
				int tmp_sum = 0;
				tmp_sum += bank[0]->subarray[3]->cells[0][i - col_mid] * 1;
				for (j = row_mid; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[3]->cells[j - row_mid + 1][i - col_mid] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			tos[0]->read_row_for_all++;
		}
		// merge output for parity
		for (int i = 0; i < w * m; i++)
		{
			for (int j = 0; j < 64; j++)
				merge[i + w * k][j] = output[i][j];
		}
		// flush to disk
		Flush_to_disk_combination(merge);
		tos[0]->writeback_status++;
	}
	printf("Combination Finish!\n");
}

void Flush_to_disk_combination(int merge[1024][64])
{
	// flush to disk
	for (int i = 0; i < (m + k); i++)
	{
		char diskname[128];
		sprintf(diskname, "./Disk_data_combination/Disk_data_combination_%d", i);
		ofstream file(diskname, fstream::app);
		// content
		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (merge[j + i * w][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
}

void Constr_multi_fail_decode_file_combination()
{
	// 随机选择m个文件作为故障文件
	vector<int> erased(m);
	vector<int> is_disk_failed(k + m);
	for (int i = 0; i < k + m; ++i)
		is_disk_failed[i] = 0;
	cout << "Failed disks is ";
	for (int i = 0; i < m;)
	{
		erased[i] = lrand48() % (k + m);
		if (is_disk_failed[erased[i]] == 0)
		{
			is_disk_failed[erased[i]] = 1;
			printf("%d ", erased[i]);
			i++;
		}
	}
	cout << endl;

	//cout << " use disks is ";
	vector<int> disk_used(k); // 记录decode用的是哪些块的数据
	for (int i = 0, index = 0; i < k; index++)
	{
		if (is_disk_failed[index] == 1)
		{
			continue;
		}
		disk_used[i] = index;
		//cout << index << " ";
		i++;
	}
	//cout << endl;

	//将未发生故障的数据块和校验块写入同一个文件中作为decode的输入
	ofstream decode_data_file("./Decode_disk_data_combination/decode_data_small_combination.file", fstream::app);
	vector<int> open_disk_file(k);
	char diskname[128];
	char buf[stripe];
	for (int i = 0; i < k; i++)
	{ // 打开要读取的文件
		sprintf(diskname, "./Disk_data_combination/Disk_data_combination_%d", disk_used[i]);
		open_disk_file[i] = open(diskname, O_RDONLY);
	}

	while (1)
	{ // 每次循环依次从disk读取w * 8字节，直到disk读完
		int ret;
		for (int i = 0; i < k; i++)
		{
			for (int num = 0; num < w; num++)
			{ // 每个文件一次读取w行, 一行为8字节
				ret = read(open_disk_file[i], buf, 8);
				for (int j = 0; j < ret; j++)
				{
					decode_data_file << buf[j];
				}
			}
		}
		if (ret == 0)
			break; //因为每个disk中字节数都一样，所以会一同完成读取
	}
	for (int i = 0; i < k; i++)
	{ // 关闭文件
		close(open_disk_file[i]);
	}
	Init_crs_in_mul_fail_combination(disk_used);
}

void Init_crs_in_mul_fail_combination(vector<int> disk_used)
{
	int *matrix, *bitmatrix, *decode_matrix, *inv_decode_matrix;
	matrix = cauchy_original_coding_matrix(k, m, w);
	// concluent CV matrix
	for (int i = 0; i < k; i++)
	{
		matrix[i] = 1;
	}
	decode_matrix = new int[k * k];
	inv_decode_matrix = new int[k * k];
	int i = 0, j = 0;
	for (i = 0; i < k; i++)
	{
		if (disk_used[i] < k)
		{
			for (j = 0; j < k; j++)
			{ // 选择的数据块在单位矩阵中的对应行
				if (j == disk_used[i])
					decode_matrix[i * k + j] = 1;
				else
					decode_matrix[i * k + j] = 0;
			}
		}
		else
		{
			for (j = 0; j < k; j++)
			{ // 选择的校验块对应的编码矩阵行
				decode_matrix[i * k + j] = matrix[(disk_used[i] - k) * k + j];
			}
		}
	}

	int tmp = jerasure_invert_matrix(decode_matrix, inv_decode_matrix, k, w);
	bitmatrix = jerasure_matrix_to_bitmatrix(k, k, w, inv_decode_matrix);

	// 映射矩阵
	int row_mid = w * k / 2;
	int col_mid = k * w / 2;
	int array_num = PIM_InputParameter->matrix;

	for (i = 0; i < col_mid; i++)
	{
		for (j = 0; j < row_mid; j++)
		{
			bank[0]->subarray[0]->cells[j + 1][i] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = col_mid; i < w * m; i++)
	{
		for (j = 0; j < row_mid; j++)
		{
			bank[0]->subarray[1]->cells[j + 1][i - col_mid] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = 0; i < col_mid; i++)
	{
		for (j = row_mid; j < w * k; j++)
		{
			bank[0]->subarray[2]->cells[j - row_mid + 1][i] = bitmatrix[i * (w * k) + j];
		}
	}

	for (i = col_mid; i < w * m; i++)
	{
		for (j = row_mid; j < w * k; j++)
		{
			bank[0]->subarray[3]->cells[j - row_mid + 1][i - col_mid] = bitmatrix[i * (w * k) + j];
		}
	}
	//printf("Finish mapping decode matrix\n");
}

void Decode_multi_fail_combination()
{
	Constr_multi_fail_decode_file_combination();
	int fp = open("./Decode_disk_data_combination/decode_data_small_combination.file", O_RDONLY);

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{ // 读取文件并转为二进制
			int ret = read(fp, buf, 8);
			tos[0]->fetch_status_times++;
			if (ret == 0)
			{
				finish_read = true;
				break;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
		}
		if(finish_read) break;
		// decode data
		int output[stripe][64];
		int i, j;
		int row_mid = w * k / 2;
		int col_mid = k * w / 2;
		for (int h = 0; h < 64; h++)
		{
			for (i = 0; i < col_mid; i++)
			{
				int tmp_sum = 0;
				for (j = 0; j < row_mid; j++)
				{
					tmp_sum += bank[0]->subarray[0]->cells[j + 1][i] * input[j][h];
				}
				bank[0]->subarray[2]->cells[0][i] = tmp_sum % 2;
			}
			for (i = col_mid; i < w * m; i++)
			{
				int tmp_sum = 0;
				for (j = 0; j < row_mid; j++)
				{
					tmp_sum += bank[0]->subarray[1]->cells[j + 1][i - col_mid] * input[j][h];
				}
				bank[0]->subarray[3]->cells[0][i - col_mid] = tmp_sum % 2;
			}
			for (i = 0; i < col_mid; i++)
			{
				int tmp_sum = 0;
				tmp_sum += bank[0]->subarray[2]->cells[0][i] * 1;
				for (j = row_mid; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[2]->cells[j - row_mid + 1][i] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			for (i = col_mid; i < w * m; i++)
			{
				int tmp_sum = 0;
				tmp_sum += bank[0]->subarray[3]->cells[0][i - col_mid] * 1;
				for (j = row_mid; j < w * k; j++)
				{
					tmp_sum += bank[0]->subarray[3]->cells[j - row_mid + 1][i - col_mid] * input[j][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			tos[0]->read_row_for_all++;
		}
		// merge output for parity
		for (int i = 0; i < w * k; i++)
		{
			for (int j = 0; j < 64; j++)
				merge[i][j] = output[i][j];
		}
		// flush to disk
		Flush_to_decode_disk_combination(merge);
		tos[0]->writeback_status++;
	}
	//printf("Finish!\n");
}

void Flush_to_decode_disk_combination(int merge[1024][64]){
	for (int i = 0; i < k; i++)
	{
		char diskname[128];
		sprintf(diskname, "./Decode_disk_data_combination/Decode_disk_data_combination_%d", i);
		ofstream file(diskname, fstream::app);
		// content
		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (merge[j + i * w][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
}

int Constr_Sing_fail_decode_file_combination()
{
	int failed_disk = lrand48() % k;
	cout << "Failed disks is " << failed_disk << endl;
	//cout << " use disks is ";
	vector<int> disk_used(k); // 记录decode用的是哪些块的数据
	for (int i = 0, index = 0; i < k; index++)
	{
		if (index == failed_disk)
		{
			continue;
		}
		disk_used[i] = index;
		//cout << index << " ";
		i++;
	}
	//cout << endl;

	//将未发生故障的数据块和校验块写入同一个文件中作为decode的输入
	ofstream decode_data_file("./Decode_disk_data_combination/decode_singal_data_small_combination.file", fstream::app);
	vector<int> open_disk_file(k);
	char diskname[128];
	char buf[stripe];
	for (int i = 0; i < k; i++)
	{ // 打开要读取的文件
		sprintf(diskname, "./Disk_data_combination/Disk_data_combination_%d", disk_used[i]);
		open_disk_file[i] = open(diskname, O_RDONLY);
	}

	while (1)
	{
		int ret;
		for (int i = 0; i < k; i++)
		{
			for (int num = 0; num < w; num++)
			{ // 每个文件一次读取w行, 一行为8字节
				ret = read(open_disk_file[i], buf, 8);
				for (int j = 0; j < ret; j++)
				{
					decode_data_file << buf[j];
				}
			}
		}
		if (ret == 0)
			break; //当前磁盘完成读取
	}
	for (int i = 0; i < k; i++)
	{ // 关闭文件
		close(open_disk_file[i]);
	}
	return failed_disk;
}

void Decode_Sing_fail_combination()
{
	int failed_disk = Constr_Sing_fail_decode_file_combination();

	int fp = open("./Decode_disk_data_combination/decode_singal_data_small_combination.file", O_RDONLY);
	char diskname[128];
	sprintf(diskname, "./Decode_disk_data_combination/Decode_sing_disk_data_combination_%d", failed_disk);
	remove(diskname);

	char buf[stripe];
	int array_num = PIM_InputParameter->matrix;
	int merge[1024][64];
	int input[stripe][64];
	int count;
	bool finish_read = false;
	while (!finish_read)
	{
		for (count = 0; count < w * k; count++)
		{ // 读取文件并转为二进制
			int ret = read(fp, buf, 8);
			if (ret == 0)
			{
				finish_read = true;
				break;
			}
			if (ret < 8)
			{
				int h = ret;
				for (; h < 8; h++)
					buf[h] = '0';
			}
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0, h = 7; j < 8; h--, j++)
				{
					if ((buf[i] >> (h)) & 0x01 == 1)
						input[count][i * 8 + j] = 1;
					else
						input[count][i * 8 + j] = 0;
				}
			}
		}

		if(finish_read) break;
		
		// decode data
		int output[stripe][64];
		for (int h = 0; h < 64; h++)
		{
			for (int i = 0; i < w; i++)
			{
				int tmp_sum = 0;
				for (int j = 0; j < k; j++)
				{
					tmp_sum += input[j * w + i][h];
				}
				output[i][h] = tmp_sum % 2;
			}
			//tos[0]->fetch_status_times++;
			tos[0]->read_row_for_all++;
		}

		// flush to disk
		tos[0]->writeback_status++;

		ofstream file(diskname, fstream::app);

		char buf[stripe];
		for (int j = 0; j < w; j++)
		{
			for (int l = 0; l < 8; l++)
			{
				char tmp = '\0';
				for (int h = 0; h < 8; h++)
				{
					if (output[j][l * 8 + h] == 1)
						tmp = (tmp << 1) | 1;
					else
						tmp = tmp << 1;
				}
				buf[j * 8 + l] = tmp;
			}
		}
		for (int j = 0; j < w * 8; j++)
		{
			file << buf[j];
		}
		file.close();
	}
	//printf("Finish!\n");
}

