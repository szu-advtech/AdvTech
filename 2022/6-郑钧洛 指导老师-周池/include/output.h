/*
Time metric:
    1.All time
    2.Offline phase time 
    3.Online phase time
*/
#include <iostream>
#include <iomanip>
using namespace std;
#define WIDTH 60
typedef struct Output{
    float all_time;
    float offline_time;
    float online_time;
    string app_name;
    Output(float _all_time, float _offline_time, float _online_time, string _app_name){
        all_time = _all_time;
        offline_time = _offline_time;
        online_time = _online_time;
        app_name = _app_name;
    }
    void draw(){
        cout << "+";
        for(int i = 0; i < WIDTH; i++){
            cout << "-";
        }
        cout << "+";
        cout << endl;
        cout << "| Application: " << setiosflags(ios::left)<< setw(WIDTH-15) << app_name <<  " |" << endl;
        cout << "|";
        for(int i = 0; i < WIDTH; i++){
            cout << "-";
        }
        cout << "|" << endl;
        cout << "|" << setiosflags(ios::left) << setw(16) << " All Time"  << setiosflags(ios::left) << setw(24) <<  "| Offline Phase Time" << setiosflags(ios::left) << setw(23) << "| Online Phase Time |" << endl; 
        cout << "|"  << " "  << setiosflags(ios::left) << setw(15) << setiosflags(ios::fixed) << setprecision(2) << all_time << "| " << setiosflags(ios::left) << setw(22)  << setiosflags(ios::fixed) << setprecision(2) << offline_time << "| " << setiosflags(ios::left) << setw(18) << setiosflags(ios::fixed) << setprecision(2) << online_time << "|" << endl; 
        cout << "+";
        for(int i = 0; i < WIDTH; i++){
            cout << "-";
        }
        cout << "+" << endl;
    }
}Output;