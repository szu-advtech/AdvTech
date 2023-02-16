/*
This is the source code of Linear regression
*/
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
#include "../include/ParSecureML.h"
#include "../include/read.h"
#include "../include/output.h"
#include "mpi.h"
using namespace std;

float *labels;
float *images;
string imagesPath;
string labelsPath;
int isRand = 1;
int train_size;
int batch_size;
int SIZE;
int row;
int col;
int FSIZE;
float alpha;

int dim1 = 784;
int dim2 = 784;
int dim3 = 64;
int dim4 = 10;
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
    string setupFile = argv[1];
    ifstream setup(setupFile);
    if(!setup.is_open()){
        cout << "Setup not exist." << endl;
        return 0;
    }
    string line;
    while(getline(setup, line)){
        stringstream para(line);
        string attr;
        para >> attr;
        if(attr == "--imagesPath"){
            para >> imagesPath;
        }
        else if(attr == "--labelsPath"){
            para >> labelsPath;
        }
        else if(attr == "--train_size"){
            para >> train_size;
        }
        else if(attr == "--batch_size"){
            para >> batch_size;
        }
        else if(attr == "--SIZE"){
            para >> SIZE;
        }
        else if(attr == "--row"){
            para >> row;
        }
        else if(attr == "--col"){
            para >> col;
        }
        else if(attr == "--FSIZE"){
            para >> FSIZE;
        }
        else if(attr == "--alpha"){
            para >> alpha;
        }
        else{
            cout << "Setup parameters error. " << attr << endl;
            return 0;
        }
    }
    setup.close();
    
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


        if((images = (float*)malloc(sizeof(float)*train_size*SIZE)) == NULL){
            cout << "Malloc error." << endl;
            return 0;
        }
        if((labels = (float*)malloc(sizeof(float)*train_size)) == NULL){
            cout << "Malloc error." << endl;
            return 0;
        }
        if(isRand == 1){
            Generator(images, 256,  train_size*SIZE);
            Generator(labels, 10, train_size);
        }else{
            read_Images(imagesPath, images, SIZE, train_size);
            Generator(labels, 10, train_size);
        }
        float *model1 = (float*)malloc(sizeof(float)*SIZE);
        for(int i = 0; i < SIZE; i++){
            model1[i] = (float)rand()/RAND_MAX;
        }
        float *images_th1 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *images_th2 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1_th1 = (float*)malloc(sizeof(float)*SIZE);
        float *model1_th2 = (float*)malloc(sizeof(float)*SIZE);
        float *y_hat1 = (float*)malloc(sizeof(float)*train_size);
        float *y_hat2 = (float*)malloc(sizeof(float)*train_size);
        double c_mstime, c_metime;
        
        for(int i = 0; i < train_size; i++){
            for(int j = 0; j < SIZE; j++){
                images_th1[i*SIZE+j] = rand() % 256;
                images_th2[i*SIZE+j] = images[i*SIZE+j] - images_th1[i*SIZE+j];	
            }
        }
        
        for(int i = 0; i < SIZE; i++){
            model1_th1[i] = (float)rand() /RAND_MAX;
            model1_th2[i] = model1[i] - model1_th1[i];
        }
        for(int i = 0; i < train_size; i++){
            y_hat1[i] = rand()%10;
            y_hat2[i] = labels[i] - y_hat1[i];
        }	
        
        int flag1 = 0;
        int flag2 = -1;
        c_mstime = timestamp();
        Support sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9;
        sp1.GetShape(batch_size, SIZE, SIZE, 1);
        sp1.Initial();
        sp1.Assign();
        //cout << "here" << endl;
        sp2.GetShape(SIZE, batch_size, batch_size, 1);
        sp2.Initial();
        sp2.Assign();

        c_metime = timestamp();
        MPI_Send(&flag1, 1, MPI_INT, server1_group_rank, 0, client2server1_comm);
        MPI_Send(images_th1, train_size*SIZE, MPI_FLOAT, server1_group_rank, 0, client2server1_comm);
        MPI_Send(model1_th1, SIZE, MPI_FLOAT, server1_group_rank, 0, client2server1_comm);
        MPI_Send(y_hat1, train_size, MPI_FLOAT, server1_group_rank, 0, client2server1_comm);
        
        MPI_Send(&flag2, 1, MPI_INT, server2_group_rank, 0, client2server2_comm);
        MPI_Send(images_th2, train_size*SIZE, MPI_FLOAT, server2_group_rank, 0, client2server2_comm);
        MPI_Send(model1_th2, SIZE, MPI_FLOAT,server2_group_rank, 0, client2server2_comm);
        MPI_Send(y_hat2, train_size, MPI_FLOAT, server2_group_rank, 0, client2server2_comm);

        sp1.Send(server1_group_rank, server2_group_rank, client2server1_comm, client2server2_comm);
        sp2.Send(server1_group_rank, server2_group_rank, client2server1_comm, client2server2_comm);

        MPI_Status status;
        MPI_Recv(model1_th1, SIZE, MPI_FLOAT, server1_group_rank, 0, client2server1_comm, &status);
        MPI_Recv(model1_th2, SIZE, MPI_FLOAT, server2_group_rank, 0, client2server2_comm, &status);

        for(int i = 0; i < SIZE; i++){
            model1[i] = model1_th1[i] + model1_th2[i];
        }
	    free(images_th1);
        free(images_th2);
        free(model1_th1);
        free(model1_th2);
        free(y_hat1);
        free(y_hat2);
        free(images);
        free(labels);
        sp1.Release();
        sp2.Release();

        float server1_offline_time, server2_offline_time;
        float server1_online_time, server2_online_time;
        
        
        MPI_Recv(&server1_offline_time, 1, MPI_FLOAT, server1_group_rank,  0, client2server1_comm, &status);
        MPI_Recv(&server2_offline_time, 1, MPI_FLOAT, server2_group_rank,  0, client2server2_comm, &status);
        MPI_Recv(&server1_online_time, 1, MPI_FLOAT, server1_group_rank, 0, client2server1_comm, &status);
        MPI_Recv(&server2_online_time, 1, MPI_FLOAT, server2_group_rank,  0, client2server2_comm, &status);
        
        all_etime = timestamp();
        float all_time = all_etime - all_stime;
        Output out(
            all_time,
            server1_offline_time > server2_offline_time ? server1_offline_time : server2_offline_time,
            server1_online_time > server2_online_time ? server1_online_time : server2_online_time,
            "Logistic regression" 
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
        float y_hat[train_size];
        float *img = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1 = (float*)malloc(sizeof(float)*SIZE*50);
        
        MPI_Recv(&flag, 1, MPI_INT, client_group_rank, 0, client2server_comm, &status);
        MPI_Recv(img, train_size*SIZE, MPI_FLOAT, client_group_rank, 0, client2server_comm, &status);
        MPI_Recv(model1, SIZE, MPI_FLOAT, client_group_rank, 0, client2server_comm, &status);
        MPI_Recv(y_hat, train_size, MPI_FLOAT, client_group_rank, 0, client2server_comm, &status);

        Triplet T1, T2;
        T1.GetShape(batch_size, SIZE, SIZE, 1);
        T1.Initial();
        T2.GetShape(SIZE, batch_size, batch_size, 1);
        T2.Initial();
        T1.Recv(client_group_rank, client2server_comm);
        T2.Recv(client_group_rank, client2server_comm);

        offline_etime = timestamp();
        double online_stime, online_etime;
        
        float batchY[batch_size];
        float *batchImg = (float*)malloc(sizeof(float)*batch_size*SIZE);
        online_stime = timestamp();
        for(int i = 0; i+batch_size < train_size; i+=batch_size){
            SelectBatch(img, y_hat, batchImg, batchY, i, batch_size, SIZE);
            T1.GetData(batchImg, model1);
            T1.Rec(server_group_rank, server2server_comm);
            T1.OP(flag);
            T1.Activation(server_group_rank, server2server_comm);
            for(int j = 0; j < batch_size; j++){
                T1.C[j] = batchY[j] - T1.C[j];
            }
            T2.GetData(batchImg, T1.C);
            T2.Rec(server_group_rank, server2server_comm);
            T2.OP(flag);
            for(int j = 0; j < SIZE; j++){
                model1[j] += alpha/batch_size*T2.C[j];
            }
        }
        online_etime = timestamp();

        MPI_Send(model1, SIZE, MPI_FLOAT, client_group_rank, 0, client2server_comm);

        T1.Release();
        T2.Release();
        free(img);
        free(model1); 

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
