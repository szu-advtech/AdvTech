/*
This file contains some function for loading data from storage
*/
#include <omp.h>
#include <iostream>
#include <cstring>
#include "../include/read.h"
#include <fstream>
#include <ostream>
#include <istream>
#include "../include/ParSecureML.h"
/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
extern int intRand(const int & min, const int & max, bool lastUsed = false);

void read_Images(string filename, float *images, int SIZE, int train_size){
    ifstream file(filename, ios::in);
    if(!file.is_open()){
        cout << "Open file failed." << endl;
    }
    for(int i = 0; i < train_size*SIZE; i++){
        file >> images[i];
    }
    file.close();
}
void read_Label(string filename, float *labels, int train_size){
    ifstream file(filename, ios::in);
    if(!file.is_open()){
        cout << "Open file failed." << endl;
    }
    for(int i = 0; i < train_size; i++){
        file >> labels[i];
    }
    file.close();
}

/*
generate random input to evaluation speed
*/
void Generator(float *array, int bound, int SIZE){
#pragma omp parallel for
    for(int i = 0; i < SIZE; i++){
        array[i] = intRand(0, bound);
    }
}
