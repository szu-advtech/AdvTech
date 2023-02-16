#include <iostream>
#include <string.h>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include <istream>
using namespace std;
void read_Images(string filename, float *images, int SIZE, int train_size);
void read_Label(string filename, float *labels, int train_size);
void Generator(float *array, int bound, int SIZE);