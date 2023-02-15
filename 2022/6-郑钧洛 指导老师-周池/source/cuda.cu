/*
This file contains some function for GPU operation
*/
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../include/ParSecureML_noMPI.h"
/*
Malloc GPU memory
*/
void MallocD(float *&gpu_a, int size){
    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&gpu_a, sizeof(*gpu_a)*size);
    if(cudaStat != cudaSuccess){
        cout << "Malloc failed:" << cudaGetErrorString(cudaStat)<< endl;
        exit(0);
    }
}
/*
Transfer data from host to device
*/
void CopyHtoD(float *gpu_a, float *a, int size){
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(gpu_a, a, sizeof(*a)*size, cudaMemcpyHostToDevice);
    if(cudaStat != cudaSuccess){
        cout << "Error code:" << cudaStat << endl;
        cout << "CopyHtoD failed." << endl;
        exit(0);
    }
}

/*
Transfer data from device to host
*/
void CopyDtoH(float *&a, float *&gpu_a, int size){
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(a, gpu_a, sizeof(*a)*size, cudaMemcpyDeviceToHost);
    if(cudaStat != cudaSuccess){
        cout << "Error code:" << cudaStat << endl;
        cout << "CopyDtoH failed." << endl;
        exit(0);
    }
}

/*
Matrix multiplication for Support class
*/
void Support_noMPI::GPU_Mul(){
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat = cublasCreate(&handle);
    stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    if(stat != CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS create failed." << endl;
        exit(0);
    }
    float alpha = 1;
    float b = 0;
    
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row1, col2, col1, &alpha, GPU_U, row1, GPU_V, row2, &b, GPU_Z, row1);
    if(stat != CUBLAS_STATUS_SUCCESS){
        cout << "Cublas sgemm failed." << endl;
        exit(0);
    }
}

void ReleaseGPU(float *A){
    cudaFree(A);
}

/*
Triplet sum function
sum = fac1+fac2+fac3
*/
__global__ void cudaTripletSum(float *sum, float *fac1, float *fac2, float *fac3,  int size){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cur = bid*blockDim.x+tid;
    if(cur >= size) return;
    float tmp = fac1[cur] + fac2[cur] + fac3[cur];
    sum[cur] = tmp;
}

/*
sum = A + B
*/
__global__ void cudaSum(float *A, float *B, float *sum, int size){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cur = bid*blockDim.x+tid;
    if(cur >= size) return;
    float tmp = A[cur]+B[cur];
    sum[cur] = tmp;
}

/*
min = A - B
*/
__global__ void cudaMinus(float *A, float *B, float *min, int size){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cur = bid*blockDim.x+tid;
    if(cur >= size) return;
    float tmp = A[cur]-B[cur];
    min[cur] = tmp;
}

/*
Triplet multiplication
if server1:
    result = A*F + E*B + Z
if server2:
    result = (A-E)*F + E*B + Z
*/
void Triplet_noMPI::cudaTripletMul(int flag){
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaError_t cudaStat;
    stat = cublasCreate(&handle);
    stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    if(stat != CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS create failed." << endl;
        exit(0);
    }
    float alpha1 = 1;
    float alpha2 = 1;
    float b = 0;
    while(flag1 == 0){
        continue;
    }
    if(flag == 0){
        while(flag2 == 0){
            continue;
        }
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row1, col2, col1, &alpha1, GPU_A, row1, GPU_F, row2, &b, fac1, row1);
        if(stat != CUBLAS_STATUS_SUCCESS){
            cout << "Cublas sgemm failed." << endl;
            exit(0);
        }
    }
    else if(flag == 1){
        cudaMinus<<<row1*col1/THREADNUM+1, THREADNUM>>>(GPU_A, GPU_E, GPU_D, row1*col1);
        while(flag2 == 0){
            continue;
        }
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row1, col2, col1, &alpha1, GPU_D, row1, GPU_F, row2, &b, fac1, row1);
        if(stat != CUBLAS_STATUS_SUCCESS){
            cout << "Cublas sgemm failed." << endl;
            exit(0);
        }
    }
    while(flag3 == 0){
        continue;
    }
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row1, col2, col1, &alpha2, GPU_E, row1, GPU_B, row2, &b, fac2, row1);
    if(stat != CUBLAS_STATUS_SUCCESS){
        cout << "Cublas sgemm failed." << endl;
        exit(0);
    }
    cublasDestroy(handle);
    cudaTripletSum<<<row1*col1/THREADNUM+THREADNUM, THREADNUM>>>(GPU_C, fac1, fac2, GPU_Z, row1*col2);
    cudaStat = cudaGetLastError();
    if(cudaStat != cudaSuccess){
        cout << "Kernel launch failed." << endl;
        exit(0);
    }
}

/*
Convolution for SecureML on GPU
*/
__global__ void cudaConv(int flag, float *GPU_A, float *GPU_B, float *GPU_C, float *GPU_E, float *GPU_F, float *GPU_Z, int row1, int col1, int row2, int col2, int o_row, int o_col, int num){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cur = bid*blockDim.x+tid;
    if(cur >= num*o_row*o_col) return;
    int num_cur = cur/(o_row*o_col);
    int row_cur = cur%(o_row*o_col)/o_col;
    int col_cur = cur%(o_row*o_col)%o_col;
    float tem = 0;
    for(int i = 0; i < row2; i++){
        for(int j = 0; j < col2; j++){
            tem += flag*GPU_E[num_cur*o_row*o_col*row2*col2+row_cur*o_col*row2*col2+col_cur*row2*col2+i*col2*j]*GPU_F[i*col2+j] + GPU_A[num_cur*row1*col1+(row_cur+i)*row1+col_cur+j] * GPU_F[i*col2+j] + GPU_E[num_cur*o_row*o_col*row2*col2+row_cur*o_col*row2*col2+col_cur*row2*col2+i*col2*j] * GPU_B[i*col2+j] + GPU_Z[i*col2+j];
        }
    }
    GPU_C[num_cur*o_row*o_col+row_cur*o_col+col_cur] = tem;
}
void ConvTriplet_noMPI::GPU_OP(int flag){
    cudaConv<<<num*o_row*o_col/THREADNUM+1, THREADNUM>>>(flag, GPU_A, GPU_B, GPU_C, GPU_E, GPU_F, GPU_Z, row1, col1, row2, col2, o_row, o_row, num);
}
