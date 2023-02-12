#include <iostream>
#include <string.h>
struct CSR{
	float *val;
	int *col;
	int *row;
};
void MallocCsr(struct CSR &C, int val_count, int row, int col);
void ReleaseCsr(struct CSR &C);
void Compress(float *matrix, struct CSR &C, int val_count, int row, int col);
void deCompress(struct CSR &C, int val_count, int row, int col, float *mat);