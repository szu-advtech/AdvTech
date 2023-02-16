#include <iostream>
#include <cstring>
#include <vector>
using namespace std;

const int MPI_client = 0;
const int MPI_server1 = 1;
const int MPI_server2 = 2;
const int CHUNK_SIZE =2;
const int BLOCKNUM = 1024;
const int THREADNUM = 1024;

class Support_noMPI{
	public:
		float *U1;
		float *U2;
		float *U;
		float *V1;
		float *V2;
		float *V;
		float *Z;
		float *Z1;
		float *Z2;
		float *GPU_U;
		float *GPU_V;
		float *GPU_Z;
		int row1;
		int col1;
		int row2;
		int col2;
		void GetShape(int in_row1, int in_col1, int in_row2, int in_col2);
		void GPU_Mul();
		void Initial();
		void Assign();
		void Release();
};

class ConvSupport_noMPI{
	public:
		float *U1;
		float *U2;
		float *U;
		float *V1;
		float *V2;
		float *V;
		float *Z;
		float *Z1;
		float *Z2;
		int row;
		int col;
		void GetShape(int in_row, int in_col);
		void Initial();
		void Assign();
		void Release();
};
class Triplet_noMPI{
	public: 
		float *A;
		float *old_A;
		float *delta_A;
		float *B;
		float *old_B;
		float *delta_B;
		float *C;
		float *D;
		float *U;
		float *V;
		float *E1;
		float *E2;
		float *E;
		float *F1;
		float *F2;
		float *F;
		float *Z;
		int row1;
		int col1;
		int row2;
		int col2;
		float *GPU_A;
		float *GPU_B;
		float *GPU_C;
		float *GPU_E;
		float *GPU_F;
		float *GPU_Z;
		float *GPU_D;
		float *fac1;
		float *fac2;
		int flag1;
		int flag2;
		int flag3;
		int pFlag;
		double commtime;
		pthread_t pipelineId;
		void GetShape(int in_row1, int in_col1, int in_row2, int in_col2);
		void Initial();
		void Release();
		void GetData(float *input_A, float *input_B);
		void OP(int flag);
		void cudaTripletMul(int flag);
		void cpuToGPU();
		static void *threadPipeline(void *ptr);
};
class ConvTriplet_noMPI{
	public: 
		float *A;
		float *B;
		float *C;
		float *U;
		float *V;
		float *E1;
		float *E2;
		float *E;
		float *F1;
		float *F2;
		float *F;
		float *Z;
		float *GPU_A;
		float *GPU_B;
		float *GPU_E;
		float *GPU_F;
		float *GPU_Z;
		float *GPU_C;
		int row1;
		int col1;
		int row2;
		int col2;
		int num;
		int o_row;
		int o_col;
		float *convA;
		double commtime;
		void GetShape(int in_row1, int in_col1, int in_row2, int in_col2, int in_num);
		void Initial();
		void Release();
		void GetData(float *input_A, float *input_filter);
		void OP(int flag);
		void GPU_OP(int flag);
};
void SelectBatch(float *Img, float *y_hat, float *batchImg, float *batchY, int cur, int batch_size, int SIZE);
void MallocD(float *&gpu_a, int size);
void CopyHtoD(float *gpu_a, float *a, int size);
void CopyDtoH(float *&a, float *&gpu_a, int size);
void ReleaseGPU(float *A);
double timestamp();





