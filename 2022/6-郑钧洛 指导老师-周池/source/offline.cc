#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include "../include/ParSecureML.h"
#include "mpi.h"

/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
int intRand(const int & min, const int & max, bool lastUsed = false) {
    static thread_local mt19937* generator = nullptr;
    hash<thread::id> hasher;
    if (!generator) generator = new mt19937(clock() + hasher(this_thread::get_id()));
    uniform_int_distribution<int> distribution(min, max);
    if (lastUsed){
        int returnVal = distribution(*generator);
        if (generator) delete(generator);
        return returnVal;
    }
    return distribution(*generator);
}

void Support_noMPI::GetShape(int in_row1, int in_col1, int in_row2, int in_col2){
	row1 = in_row1;
	col1 = in_col1;
	row2 = in_row2;
	col2 = in_col2;
}
/*
Full connect layer shape:
	U,U1,U2:row1*col1
	V,V1,V2:row2*col2
	Z,Z1,Z2:row1*col2
*/
void Support_noMPI::Initial(){
	U1 = (float*)malloc(sizeof(float)*row1*col1);
	U2 = (float*)malloc(sizeof(float)*row1*col1);
	U = (float*)malloc(sizeof(float)*row1*col1);
	V1 = (float*)malloc(sizeof(float)*row2*col2);
	V2 = (float*)malloc(sizeof(float)*row2*col2);
	V = (float*)malloc(sizeof(float)*row2*col2);
	Z1 = (float*)malloc(sizeof(float)*row1*col2);
	Z2 = (float*)malloc(sizeof(float)*row1*col2);
	Z = (float*)malloc(sizeof(float)*row1*col2);
	MallocD(GPU_U, row1*col1);
	MallocD(GPU_V, row2*col2);
	MallocD(GPU_Z, row1*col2);
}

/*
Client:
	U = U1+U2
	V = V1+V2
	Z = U*V
	Z = Z1+Z2
*/
void Support_noMPI::Assign(){
#pragma omp parallel
{
#pragma omp for
    for(int i = 0; i < row1*col1; i++){
		U1[i] = (float)intRand(0, RAND_MAX);
		U2[i] = (float)intRand(0, RAND_MAX);
		U[i] = U1[i] + U2[i];
	}
#pragma omp for
	for(int i = 0; i < row2*col2; i++){
		V1[i] = (float)intRand(0, RAND_MAX);
		V2[i] = (float)intRand(0, RAND_MAX);
		V[i] = V1[i] + V2[i];
	}
}

	CopyHtoD(GPU_U, U, row1*col1);
	CopyHtoD(GPU_V, V, row2*col2);
	GPU_Mul();
	CopyDtoH(Z, GPU_Z, row1*col2);
#pragma omp parallel
{
#pragma omp for
    for(int i = 0; i < row1*col2; i++){
		Z1[i] = (float)intRand(0, RAND_MAX);
		Z2[i] = Z[i] - Z1[i];
	}
}
    
}


/*
Send data to server1 and server2
@parms: 
	MPI_dest1, MPI_dest2: MPI_rank of server1 and server2
	server1_comm, server2_comm: MPI_comm of server1 and server2
*/

void Support::Send(int MPI_dest1, int MPI_dest2, MPI_Comm server1_comm, MPI_Comm server2_comm){
	MPI_Send(U1, row1*col1, MPI_FLOAT, MPI_dest1, 0, server1_comm);
	MPI_Send(V1, row2*col2, MPI_FLOAT, MPI_dest1, 0, server1_comm);
	MPI_Send(Z1, row1*col2, MPI_FLOAT, MPI_dest1, 0, server1_comm);

	MPI_Send(U2, row1*col1, MPI_FLOAT, MPI_dest2, 0, server2_comm);
	MPI_Send(V2, row2*col2, MPI_FLOAT, MPI_dest2, 0, server2_comm);
	MPI_Send(Z2, row1*col2, MPI_FLOAT, MPI_dest2, 0, server2_comm);
}

void Support_noMPI::Release(){
	free(U1);
	free(U2);
	free(U);
	free(V1);
	free(V2);
	free(V);
	free(Z);
    free(Z1);
    free(Z2);
    ReleaseGPU(GPU_U);
    ReleaseGPU(GPU_V);
    ReleaseGPU(GPU_Z);
}
void ConvSupport_noMPI::GetShape(int in_row, int in_col){
	row = in_row;
	col = in_col;
}

/*
Convolution layer shape:
	U,U1,U2,V,V1,V2,Z,Z1,Z2: row*col
*/
void ConvSupport_noMPI::Initial(){
	U1 = (float*)malloc(sizeof(float)*row*col);
	U2 = (float*)malloc(sizeof(float)*row*col);
	U = (float*)malloc(sizeof(float)*row*col);
	V1 = (float*)malloc(sizeof(float)*row*col);
	V2 = (float*)malloc(sizeof(float)*row*col);
	V = (float*)malloc(sizeof(float)*row*col);
	Z1 = (float*)malloc(sizeof(float)*row*col);
	Z2 = (float*)malloc(sizeof(float)*row*col);
	Z = (float*)malloc(sizeof(float)*row*col);
}

void ConvSupport_noMPI::Assign(){
#pragma omp parallel
{
#pragma omp for
    for(int i = 0; i < row*col; i++){
		U1[i] = (float)intRand(0, RAND_MAX);
		U2[i] = (float)intRand(0, RAND_MAX);
		U[i] = U1[i] + U2[i];
	}
#pragma omp for
	for(int i = 0; i < row*col; i++){
		V1[i] = (float)intRand(0, RAND_MAX);
		V2[i] = (float)intRand(0, RAND_MAX);
		V[i] = V1[i] + V2[i];
	}
#pragma omp for
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			Z[i*col+j] = U[i*col+j] * V[i*col+j];
		}
	}
#pragma omp for
    for(int i = 0; i < row*col; i++){
		Z1[i] = (float)intRand(0, RAND_MAX);
		Z2[i] = Z[i] - Z1[i];
	}
}

}

/*
Send data function of Convolution support
*/
void ConvSupport::Send(int MPI_dest1, int MPI_dest2, MPI_Comm server1_comm, MPI_Comm server2_comm){
	MPI_Send(U1, row*col, MPI_FLOAT, MPI_dest1, 0, server1_comm);
	MPI_Send(V1, row*col, MPI_FLOAT, MPI_dest1, 0, server1_comm);
	MPI_Send(Z1, row*col, MPI_FLOAT, MPI_dest1, 0, server1_comm);

	MPI_Send(U2, row*col, MPI_FLOAT, MPI_dest2, 0, server2_comm);
	MPI_Send(V2, row*col, MPI_FLOAT, MPI_dest2, 0, server2_comm);
	MPI_Send(Z2, row*col, MPI_FLOAT, MPI_dest2, 0, server2_comm);
}

void ConvSupport_noMPI::Release(){
#pragma omp parallel
{
    intRand(0, 1, true);
}
	free(U1);
	free(U2);
	free(U);
	free(V1);
	free(V2);
	free(V);
	free(Z);
    free(Z1);
    free(Z2);
}
