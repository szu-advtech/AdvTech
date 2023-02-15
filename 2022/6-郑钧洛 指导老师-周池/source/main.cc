#include <omp.h>
#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <random>
#include <thread>
#include "../include/ParSecureML.h"
#include "../include/read.h"
#include "mpi.h"
#include "../include/output.h"
using namespace std;

#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif

#define FLOAT_PER_CACHELINE (64/4)
const int N = 100;


/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
extern int intRand(const int & min, const int & max, bool lastUsed = false);

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int MPI_rank, MPI_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    double all_stime, all_etime;
    all_stime = timestamp();
    double offline_stime;
    offline_stime = timestamp();
    if(MPI_rank == MPI_client){

        /*
        Init MPI group: client2server1 clientserver2
        */

        const int group_client2server1[CHUNK_SIZE] = {MPI_client, MPI_server1};
        const int group_client2server2[CHUNK_SIZE] = {MPI_client, MPI_server2};

        MPI_Group client2server1;
        MPI_Group client2server2;

        MPI_Group_incl(world_group, CHUNK_SIZE, group_client2server1, &client2server1);
        MPI_Group_incl(world_group, CHUNK_SIZE, group_client2server2, &client2server2);

        MPI_Comm client2server1_comm;
        MPI_Comm client2server2_comm;

        MPI_Comm_create_group(MPI_COMM_WORLD, client2server1, 0, &client2server1_comm);
        MPI_Comm_create_group(MPI_COMM_WORLD, client2server2, 0, &client2server2_comm);

        int MPI_group_client2server1_rank;
        int MPI_group_client2server2_rank;

        MPI_Comm_rank(client2server1_comm, &MPI_group_client2server1_rank);
        MPI_Comm_rank(client2server2_comm, &MPI_group_client2server2_rank);

        int server1_group_rank = 0, server2_group_rank = 0;

        if(MPI_group_client2server1_rank == 0){
            server1_group_rank = 1;
        }
        if(MPI_group_client2server2_rank == 0){
            server2_group_rank = 1;
        }

        float *A, *B, *C;
        A = (float*)malloc(sizeof(float)*N*N);
        B = (float*)malloc(sizeof(float)*N*N);
        C = (float*)malloc(sizeof(float)*N*N);
        Generator(A, 256, N*N);
        Generator(B, 256, N*N);
        float *A1 = (float*)malloc(sizeof(float)*N*N);
        float *A2 = (float*)malloc(sizeof(float)*N*N);
        float *B1 = (float*)malloc(sizeof(float)*N*N);
        float *B2 = (float*)malloc(sizeof(float)*N*N);
        float *C1 = (float*)malloc(sizeof(float)*N*N);
        float *C2 = (float*)malloc(sizeof(float)*N*N);


        int procNum = omp_get_num_procs();
        int *seed = (int *)malloc(sizeof(int) * procNum*2);
        for (int i = 0; i < procNum*2; i++){
            seed[i] = rand();
        }
#pragma omp parallel num_threads(2*procNum)
{
        int s = seed[omp_get_thread_num()];
#pragma omp for
    	for(int i = 0; i < N*N; i++){
            A1[i] = intRand(0, 255);
            A2[i] = A[i] - A1[i];
        }
#pragma omp for
        for(int i = 0; i < N*N; i++){
            B1[i] = intRand(0, 255);
            B2[i] = B[i] - B1[i];
        } 
}


        Support sp;
        sp.GetShape(N, N, N, N);
        sp.Initial();
        sp.Assign();
        int flag1 = 0;
        int flag2 = -1;

        MPI_Send(&flag1, 1, MPI_INT, server1_group_rank, 0, client2server1_comm);
        MPI_Send(A1, N*N, MPI_FLOAT, server1_group_rank, 0, client2server1_comm);
        MPI_Send(B1, N*N, MPI_FLOAT, server1_group_rank, 0, client2server1_comm);



        MPI_Send(&flag2, 1, MPI_INT, server2_group_rank, 0, client2server2_comm);
        MPI_Send(A2, N*N, MPI_FLOAT, server2_group_rank, 0, client2server2_comm);
        MPI_Send(B2, N*N, MPI_FLOAT, server2_group_rank, 0, client2server2_comm);


        sp.Send(server1_group_rank, server2_group_rank, client2server1_comm, client2server2_comm);

 
        MPI_Status status;

        MPI_Recv(C1, N*N, MPI_FLOAT, server1_group_rank, 0, client2server1_comm, &status);
        MPI_Recv(C2, N*N, MPI_FLOAT, server2_group_rank, 0, client2server2_comm, &status);  

#pragma omp parallel for
        for(int i = 0; i < N*N/FLOAT_PER_CACHELINE; i++){
            for (int j = 0; j < FLOAT_PER_CACHELINE; j++){
                C[i * FLOAT_PER_CACHELINE + j] = C1[i * FLOAT_PER_CACHELINE + j] + C2[i * FLOAT_PER_CACHELINE + j];
            }
        }
        free(A1);
        free(A2);
        free(B1);
        free(B2);
        free(A);
        free(B);
        free(C);
        sp.Release();
        float server1_offline_time, server2_offline_time;
        float server1_online_time, server2_online_time;
        all_etime = timestamp();
        float all_time = all_etime - all_stime;

        MPI_Recv(&server1_offline_time, 1, MPI_FLOAT, server1_group_rank,  0, client2server1_comm, &status);
        MPI_Recv(&server2_offline_time, 1, MPI_FLOAT, server2_group_rank,  0, client2server2_comm, &status);
        MPI_Recv(&server1_online_time, 1, MPI_FLOAT, server1_group_rank, 0, client2server1_comm, &status);
        MPI_Recv(&server2_online_time, 1, MPI_FLOAT, server2_group_rank,  0, client2server2_comm, &status);
        Output out(
            all_time,
            server1_offline_time > server2_offline_time ? server1_offline_time : server2_offline_time,
            server1_online_time > server2_online_time ? server1_online_time : server2_online_time,
            "Single matrix multiplication" 
        );
        out.draw();
        if(client2server1 != MPI_GROUP_NULL) MPI_Group_free(&client2server1);
        if(client2server2 != MPI_GROUP_NULL) MPI_Group_free(&client2server2);
    }
    else{

        /*
        Init MPI_group: client2server, server2server
        */

        const int group_client2server[CHUNK_SIZE] = {MPI_client, MPI_rank};
        const int group_server2server[CHUNK_SIZE] = {MPI_server1, MPI_server2};

        MPI_Group client2server;
        MPI_Group server2server;

        MPI_Group_incl(world_group, CHUNK_SIZE, group_client2server, &client2server);
        MPI_Group_incl(world_group, CHUNK_SIZE, group_server2server, &server2server);

        MPI_Comm client2server_comm;
        MPI_Comm server2server_comm;

        MPI_Comm_create_group(MPI_COMM_WORLD, client2server, 0, &client2server_comm);
        MPI_Comm_create_group(MPI_COMM_WORLD, server2server, 0, &server2server_comm);

        int MPI_group_client2server_rank;
        int MPI_group_server2server_rank;

        MPI_Comm_rank(client2server_comm, &MPI_group_client2server_rank);
        MPI_Comm_rank(server2server_comm, &MPI_group_server2server_rank);

        int client_group_rank = 0, server_group_rank = 0;

        if(MPI_group_client2server_rank == 0){
            client_group_rank = 1;
        }
        if(MPI_group_server2server_rank == 0){
            server_group_rank = 1;
        }
        

        double offline_etime;
        MPI_Status status;
        int flag;
        float *A1 = (float*)malloc(sizeof(float)*N*N);
        float *B1 = (float*)malloc(sizeof(float)*N*N);

        MPI_Recv(&flag, 1, MPI_INT, client_group_rank, 0, client2server_comm, &status);
        MPI_Recv(A1, N*N, MPI_FLOAT, client_group_rank, 0, client2server_comm, &status);
        MPI_Recv(B1, N*N, MPI_FLOAT, client_group_rank, 0, client2server_comm, &status);

        Triplet T;
        T.GetShape(N, N, N, N);
        T.Initial();
        T.Recv(client_group_rank, client2server_comm);

        offline_etime = timestamp();

        double online_stime, online_etime;
        online_stime = timestamp();
        T.GetData(A1, B1);
        T.Rec(server_group_rank, server2server_comm);
        T.OP(flag);
        online_etime = timestamp();
        MPI_Send(T.C, N*N, MPI_FLOAT, client_group_rank, 0, client2server_comm);
        T.Release();

        float online_time = online_etime-online_stime;
        float offline_time = offline_etime-offline_stime;
        MPI_Send(&offline_time, 1, MPI_FLOAT, client_group_rank, 0, client2server_comm);
        MPI_Send(&online_time, 1, MPI_FLOAT, client_group_rank, 0, client2server_comm);
        if(client2server != MPI_GROUP_NULL){
            MPI_Group_free(&client2server);
        }
        if(server2server != MPI_GROUP_NULL){
            MPI_Group_free(&server2server);
        }
    }
    if(world_group != MPI_GROUP_NULL){
        MPI_Group_free(&world_group);
    }
    MPI_Finalize();
    return 0;
}
