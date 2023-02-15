#include <iostream>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/time.h>
#include <cmath>
#include "../include/ParSecureML.h"
#include "../include/compress.h"
#include "../include/read.h"
#include <mpi.h>
double timestamp(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + 1e-6*time.tv_usec;
}
void Triplet_noMPI::GetShape(int in_row1, int in_col1, int in_row2, int in_col2){
	row1 = in_row1;
	col1 = in_col1;
	row2 = in_row2;
	col2 = in_col2;
}
/*
Malloc memory for triplet && create pipeline
*/
void Triplet_noMPI::Initial(){
	commtime = 0;
	A = (float*)malloc(sizeof(float)*row1*col1);
	old_A = (float*)malloc(sizeof(float)*row1*col1);
	delta_A = (float*)malloc(sizeof(float)*row1*col1);
	B = (float*)malloc(sizeof(float)*row2*col2);
	old_B = (float*)malloc(sizeof(float)*row2*col2);
	delta_B = (float*)malloc(sizeof(float)*row2*col2);
	C = (float*)malloc(sizeof(float)*row1*col2);
	D = (float*)malloc(sizeof(float)*row1*col1);
	U = (float*)malloc(sizeof(float)*row1*col1);
	V = (float*)malloc(sizeof(float)*row2*col2);
	Z = (float*)malloc(sizeof(float)*row1*col2);
	E1 = (float*)malloc(sizeof(float)*row1*col1);
	E2 = (float*)malloc(sizeof(float)*row1*col1);
	E = (float*)malloc(sizeof(float)*row1*col1);
	F1 = (float*)malloc(sizeof(float)*row2*col2);
	F2 = (float*)malloc(sizeof(float)*row2*col2);
	F = (float*)malloc(sizeof(float)*row2*col2);
	for(int i = 0; i < row1*col1; i++){
		E[i] = (float)rand()/RAND_MAX;
	}
	for(int i = 0; i < row2*col2; i++){
		F[i] = (float)rand()/RAND_MAX;
	}
	for(int i = 0; i < row1*col1; i++){
		old_A[i] = 0;
	}
	for(int i = 0; i < row2*col2; i++){
		old_B[i] = 0;
	}
	MallocD(GPU_A, row1*col1);
	MallocD(GPU_B, row2*col2);
	MallocD(GPU_E, row1*col1);
	MallocD(GPU_F, row2*col2);
	MallocD(GPU_Z, row1*col2);
	MallocD(fac1, row1*col2);
	MallocD(fac2, row1*col2);
	MallocD(GPU_C, row1*col2);
	MallocD(GPU_D, row1*col1);
	flag1 = 0;
	flag2 = 0;
	flag3 = 0;
	pFlag = 0;
	pthread_create(&pipelineId, NULL, threadPipeline, (void*)this);
}
void Triplet_noMPI::Release(){
	free(A);
	free(B);
	free(old_A);
	free(old_B);
	free(delta_A);
	free(delta_B);
	free(C);
    free(Z);
	free(U);
	free(V);
	free(E1);
	free(E2);
	free(E);
	free(F1);
	free(F2);
	free(F);
	free(D);
	ReleaseGPU(GPU_A);
	ReleaseGPU(GPU_B);
	ReleaseGPU(GPU_C);
	ReleaseGPU(GPU_E);
	ReleaseGPU(GPU_F);
	ReleaseGPU(GPU_Z);
	ReleaseGPU(fac1);
	ReleaseGPU(fac2);
	ReleaseGPU(GPU_D);
	pFlag = 2;
	pthread_join(pipelineId, NULL);
}
void Triplet_noMPI::GetData(float *input_A, float *input_B){
	for(int i = 0; i < row1*col1; i++){
		A[i] = input_A[i];
	}
	for(int i = 0; i < row2*col2; i++){
		B[i] = input_B[i];
	}
}

/*
Prepare E,F , which need communication between server1 and server2
*/
void Triplet::Rec(int MPI_dest, MPI_Comm server2server_comm){
	int countA = 0;
	int countB = 0;
	int countA2;
	int countB2;
	/*
	delta compression
	*/
	for(int i = 0; i < row1*col1; i++){
		delta_A[i] = A[i] - old_A[i];
		if(abs(delta_A[i]) < 0.00001){
			countA++;
		}
		old_A[i] = A[i];
	}
	for(int i = 0; i < row2*col2; i++){
		delta_B[i] = B[i] - old_B[i];
		if(abs(delta_B[i]) < 0.00001){
			countB++;
		}
		old_B[i] = B[i];
	}
	for(int i = 0; i < row1*col1; i++){
		E1[i] = A[i] - U[i];
	}
	for(int i = 0; i < row2*col2; i++){
		F1[i] = B[i] - V[i];
	}
	int csrFlagA1 = 0, csrFlagA2 = 0;
	struct CSR deltaACsr1;
	struct CSR deltaACsr2;
	MPI_Status status;
	if((double)countA/(row1*col1) > 0.75){
		csrFlagA1 = 1;
	}
	MPI_Sendrecv(&csrFlagA1, 1, MPI_INT, MPI_dest, 0, &csrFlagA2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
	if(csrFlagA1==1 && csrFlagA2==1){
		MallocCsr(deltaACsr1, countA, row1, col1);
		Compress(delta_A, deltaACsr1, countA, row1, col1);
		MPI_Sendrecv(&countA, 1, MPI_INT, MPI_dest, 0, &countA2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MallocCsr(deltaACsr2, countA2, row1, col1);
		double stime = timestamp();
		MPI_Sendrecv(deltaACsr1.val, countA, MPI_FLOAT, MPI_dest, 0, deltaACsr2.val, countA2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		MPI_Sendrecv(deltaACsr1.col, countA, MPI_INT, MPI_dest, 0, deltaACsr2.col, countA2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Sendrecv(deltaACsr1.row, row1, MPI_INT, MPI_dest, 0, deltaACsr2.row, row1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (etime-stime);
		deCompress(deltaACsr2, countA2, row1, col1, E2);
		ReleaseCsr(deltaACsr1);
		ReleaseCsr(deltaACsr2);
	}
	if(csrFlagA1==1 && csrFlagA2==0){
		MallocCsr(deltaACsr1, countA, row1, col1);
		Compress(delta_A, deltaACsr1, countA, row1, col1);
		double stime = timestamp();
		MPI_Send(&countA, 1, MPI_INT, MPI_dest, 0, server2server_comm);
		MPI_Send(deltaACsr1.val, countA, MPI_FLOAT, MPI_dest, 0, server2server_comm);
		MPI_Send(deltaACsr1.col, countA, MPI_INT, MPI_dest, 0, server2server_comm);
		MPI_Send(deltaACsr1.row, row1, MPI_INT, MPI_dest, 0, server2server_comm);
		MPI_Recv(E2, row1*col1, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (etime-stime);
		ReleaseCsr(deltaACsr1);
	}
	if(csrFlagA1==0 && csrFlagA2==1){
		MPI_Recv(&countA2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MallocCsr(deltaACsr2, countA2, row1, col1);
		double stime = timestamp();
		MPI_Recv(deltaACsr2.val, countA2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		MPI_Recv(deltaACsr2.col, countA2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Recv(deltaACsr2.row, row1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Send(E1, row1*col1, MPI_FLOAT, MPI_dest, 0, server2server_comm);
		double etime = timestamp();
		commtime += (etime-stime);
		deCompress(deltaACsr2, countA2, row1, col1, E2);
		ReleaseCsr(deltaACsr2);
	}
	if(csrFlagA1==0 && csrFlagA2==0){
		double stime = timestamp();
		MPI_Sendrecv(E1, row1*col1, MPI_FLOAT, MPI_dest, 0, E2, row1*col1, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (etime-stime);
	}

	int csrFlagB1 = 0, csrFlagB2 = 0;
	if((double)countB/(row2*col2) > 0.75){
		csrFlagB1 = 1;
	}
	MPI_Sendrecv(&csrFlagB1, 1, MPI_INT, MPI_dest, 0, &csrFlagB2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
	struct CSR deltaBCsr1;
	struct CSR deltaBCsr2;
	if(csrFlagB1==1 && csrFlagB2==1){
		MallocCsr(deltaBCsr1, countB, row2, col2);
		Compress(delta_B, deltaBCsr1, countB, row2, col2);
		MPI_Sendrecv(&countB, 1, MPI_INT, MPI_dest, 0, &countB2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MallocCsr(deltaBCsr2, countB2, row2, col2);
		double stime = timestamp();
		MPI_Sendrecv(deltaBCsr1.val, countB, MPI_FLOAT, MPI_dest, 0, deltaBCsr2.val, countB2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		MPI_Sendrecv(deltaBCsr1.col, countB, MPI_INT, MPI_dest, 0, deltaBCsr2.col, countB2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Sendrecv(deltaBCsr1.row, row2, MPI_INT, MPI_dest, 0,  deltaBCsr2.row, row2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (stime-etime);
		deCompress(deltaBCsr2, countB2, row2, col2, F2);
		ReleaseCsr(deltaBCsr1);
		ReleaseCsr(deltaBCsr2);
	}
	if(csrFlagB1==1 && csrFlagB2==0){
		MallocCsr(deltaBCsr1, countB, row2, col2);
		Compress(delta_B, deltaBCsr1, countB, row2, col2);
		MPI_Send(&countB, 1, MPI_INT, MPI_dest, 0, server2server_comm);
		double stime = timestamp();
		MPI_Send(deltaBCsr1.val, countB, MPI_FLOAT, MPI_dest, 0, server2server_comm);
		MPI_Send(deltaBCsr1.col, countB, MPI_INT, MPI_dest, 0, server2server_comm);
		MPI_Send(deltaBCsr1.row, row2, MPI_INT, MPI_dest, 0, server2server_comm);
		MPI_Recv(F2, row2*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (etime-stime);
		ReleaseCsr(deltaBCsr1);
	}
	if(csrFlagB1==0 && csrFlagB2==1){
		MPI_Recv(&countB2, 1, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MallocCsr(deltaBCsr2, countB2, row2, col2);
		MPI_Recv(deltaBCsr2.val, countB2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		double stime = timestamp();
		MPI_Recv(deltaBCsr2.col, countB2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Recv(deltaBCsr2.row, row2, MPI_INT, MPI_dest, 0, server2server_comm, &status);
		MPI_Send(F1, row2*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm);
		double etime = timestamp();
		commtime += (etime-stime);
		deCompress(deltaBCsr2, countB2, row2, col2, F2);
		ReleaseCsr(deltaBCsr2);
	}
	if(csrFlagB1==0 && csrFlagB2==0){
		double stime = timestamp();
		MPI_Sendrecv(F1, row2*col2, MPI_FLOAT, MPI_dest, 0, F2, row2*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
		double etime = timestamp();
		commtime += (etime-stime);
	}
	for(int i = 0; i < row1*col1; i++){
		E[i] = E1[i] + E2[i];
	}
	for(int i = 0; i < row2*col2; i++){
		F[i] = F1[i] + F2[i];
	}
}

/*
Transer data from CPU to GPU
*/
void Triplet_noMPI::cpuToGPU(){
	CopyHtoD(GPU_E, E, row1*col1);
	CopyHtoD(GPU_A, D, row1*col1);
	flag1 = 1;
	CopyHtoD(GPU_F, F, row2*col2);
	flag2 =1;
	CopyHtoD(GPU_B, B, row2*col2);
	flag3 = 1;
}

/*
Pipeline
*/
void* Triplet_noMPI::threadPipeline(void *ptr){
	Triplet *pt = (Triplet*)ptr;
	while(pt->pFlag != 2){ // thread kill
		while(pt->pFlag == 0){
			continue;
		}
		if(pt->pFlag == 2){
			break;
		}
		pt->cpuToGPU();	
		pt->pFlag = 0;
	}
	return pt;
}

/*
OP operation for triplet
C = triplet.op(flag)
*/
void Triplet_noMPI::OP(int flag){
	pFlag = 1;
	CopyHtoD(GPU_Z, Z, row1*col2);
	cudaTripletMul(flag);
	CopyDtoH(C, GPU_C, row1*col2);
	flag1 = 0;
	flag2 = 0;
	flag3 = 0;
}


/*
Recvive data from client
@parms: 
	MPI_dest: client rank
	MPI_Comm: MPI_Comm of client2server
*/
void Triplet::Recv(int MPI_dest, MPI_Comm client2server_comm){
	MPI_Status status;
	MPI_Recv(U, row1*col1, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
	MPI_Recv(V, row2*col2, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
	MPI_Recv(Z, row1*col2, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
}

/*
Activation function
*/
void Triplet::Activation(int MPI_dest, MPI_Comm server2server_comm){
	for(int i = 0; i < row1*col2; i++){
		if(C[i] < 0) C[i] = 0;
	}
	MPI_Status status;
	MPI_Sendrecv(C, row1*col2, MPI_FLOAT, MPI_dest, 0, C, row1*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
}

/*
convolution shape:
input: row1*col1
Convolution kernels: row2*col2
output:(row1-row2+1)*(col1*col2+1)
*/
void ConvTriplet_noMPI::GetShape(int in_row1, int in_col1, int in_row2, int in_col2, int in_num){
	row1 = in_row1;
	col1 = in_col1;
	row2 = in_row2;
	col2 = in_col2;
	num = in_num;
	o_row = row1-row2+1;
	o_col = col1-col2+1;
}
void ConvTriplet_noMPI::Initial(){
	commtime = 0;
	A = (float*)malloc(sizeof(float)*num*row1*col1);
	B = (float*)malloc(sizeof(float)*row2*col2);
	C = (float*)malloc(sizeof(float)*num*o_row*o_col);
	U = (float*)malloc(sizeof(float)*row2*col2);
	V = (float*)malloc(sizeof(float)*row2*col2);
	Z = (float*)malloc(sizeof(float)*row2*col2);
	E1 = (float*)malloc(sizeof(float)*num*o_row*o_col*row2*col2);
	E2 = (float*)malloc(sizeof(float)*num*o_row*o_col*row2*col2);
	E = (float*)malloc(sizeof(float)*num*o_row*o_col*row2*col2);
	F1 = (float*)malloc(sizeof(float)*row2*col2);
	F2 = (float*)malloc(sizeof(float)*row2*col2);
	F = (float*)malloc(sizeof(float)*row2*col2);
	MallocD(GPU_A, num*row1*col1);
	MallocD(GPU_B, row2*col2);
	MallocD(GPU_C, num*o_row*o_col);
	MallocD(GPU_E, num*o_row*o_col*row2*col2);
	MallocD(GPU_F, row2*col2);
	MallocD(GPU_Z, row1*col2);
	
}
void ConvTriplet_noMPI::Release(){
	free(A);
	free(B);
	free(C);
    free(Z);
	free(U);
	free(V);
	free(E1);
	free(E2);
	free(E);
	free(F1);
	free(F2);
	free(F);
	ReleaseGPU(GPU_A);
	ReleaseGPU(GPU_B);
	ReleaseGPU(GPU_C);
	ReleaseGPU(GPU_E);
	ReleaseGPU(GPU_F);
	ReleaseGPU(GPU_Z);
}
void ConvTriplet_noMPI::GetData(float *input_A, float *input_filter){
	for(int i = 0; i < num*row1*col1; i++){
		A[i] = input_A[i];
	}
	for(int i = 0; i < row2*col2; i++){
		B[i] = input_filter[i];
	}
}

/*
prepare E,F for convolution layer
*/
void ConvTriplet::Rec(int MPI_dest, MPI_Comm server2server_comm){
	for(int s = 0; s < num; s++){
		for(int i = 0; i < o_row; i++){
			for(int j = 0; j < o_col; j++){
				for(int k = 0; k < row2; k++){
					for(int h = 0; h < col2; h++){
						E1[s*o_row*o_col*row2*col2+i*o_col*row2*col2+j*row2*col2+k*col2+h] = A[s*row1*col1+(i+k)*col1+j+h] - U[k*col2+h];
					}
				}
			}
		}
	}
	for(int i = 0; i < row2*col2; i++){
		F1[i] = B[i] - V[i];
	}
	double stime = timestamp();
	MPI_Status status;
	MPI_Sendrecv(E1, num*o_row*o_col*row2*col2, MPI_FLOAT, MPI_dest, 0, E2, num*o_row*o_col*row2*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
	MPI_Sendrecv(F1, row2*col2, MPI_FLOAT, MPI_dest, 0, F2, row2*col2, MPI_FLOAT, MPI_dest, 0, server2server_comm, &status);
	double etime = timestamp();
	commtime += (etime-stime);
	for(int i = 0; i < num*o_row*o_col*row2*col2; i++){
		E[i] = E1[i] + E2[i];
	}
	for(int i = 0; i < row2*col2; i++){
		F[i] = F1[i] + F2[i];
	}
}
float ConvMul(int flag, float *A, float *B, float *E, float *F, float * Z, int size){
	float re = 0;
	for(int i = 0; i < size; i++){
		re += flag*E[i]*F[i] + A[i]*F[i] + E[i]*B[i] + Z[i];
	}
	return re;
}
/*
OP operatioon for convolution layer
*/
void ConvTriplet_noMPI::OP(int flag){
	CopyHtoD(GPU_A, A, num*row1*col1);
	CopyHtoD(GPU_B, B, row2*col2);
	CopyHtoD(GPU_E, E, num*o_row*o_col*row2*col2);
	CopyHtoD(GPU_F, F, row2*col2);
	CopyHtoD(GPU_Z, Z, row2*col2);
	GPU_OP(flag);
	CopyDtoH(C, GPU_C, o_row*o_col);
}

void ConvTriplet::Recv(int MPI_dest, MPI_Comm client2server_comm){
	MPI_Status status;
	MPI_Recv(U, row2*col2, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
	MPI_Recv(V, row2*col2, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
	MPI_Recv(Z, row2*col2, MPI_FLOAT, MPI_dest, 0, client2server_comm, &status);
}


/*
Select a batch of data
*/
void SelectBatch(float *Img, float *y_hat, float *batchImg, float *batchY, int cur, int batch_size, int SIZE){
	for(int i = cur; i < cur+batch_size; i++){
		for(int j = 0; j < SIZE; j++){
			batchImg[(i-cur)*SIZE+j] = Img[i*SIZE+j];
		}
	}
	for(int i = cur; i < cur+batch_size; i++){
		batchY[i-cur] = y_hat[i];
	}
}
