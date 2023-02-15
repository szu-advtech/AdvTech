#include <vector>
#include "trs_tree.h"
#include <cstdio>
#include <string>
#include <fstream>
#include <iomanip>
#include "bplustree.cpp"
#include <random>
#define Log std::cout
#include <ctime>
using namespace std;

vector<string> colA;
vector<string> colB;
vector<string> colC;

namespace hermit {
  TRS::TRS_Tree* trs;

  inline void build_trs(vector<vector<string> >& table) {
//    printf("开始构建TRS-Tree\n");
    trs = new TRS::TRS_Tree();
    vector<vector<int> > _table;
    for (vector<string>& vs : table) {
      vector<int> vi({ atoi(vs[0].data()), atoi(vs[1].data()), atoi(vs[2].data()) });
      _table.emplace_back(std::move(vi));
    }
    trs -> build(_table);
  }

  inline pair<set<string>, set<string> > lookup(const string& _key, const string& _key_2) {
    // TODO: due to demo case, assume income param size() = 1
    int key = atoi(_key.data());
    int key2 = atoi(_key_2.data());
    pair<int, int> k(key, key2);
    auto p = trs -> lookup(k);
    set<string> p1;
    set<string> p2;
    for (int num : p.first) {
      p1.insert(to_string(num));
    }
    for (int num : p.second) {
      p2.insert(to_string(num));
    }
    return { p1, p2 };
  }

}

int RandomCreatFunc(int interval_min, int interval_max)
{
    if (interval_min >= interval_max)
        return INT_MAX;
    //种子值是通过 random_device 类型的函数对象 rd 获得的。
    //每一个 rd() 调用都会返回不同的值，而且如果我们实现的 random_devic 是非确定性的，程序每次执行连续调用 rd() 都会产生不同的序列。
    random_device rd;
    default_random_engine e{ rd() };
    //部分环境中random_device不可用，原因尚不知，笔者在测试过程中发现，如果不可用该用以下方式
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // default_random_engine e{seed};
    int random_num = 0;
    //分情况产生随机数，考虑3种情况：min<0&&max>0 、max<0和min>0
    if (interval_min >= 0)//min>0的情况
    {
        uniform_int_distribution<unsigned> a(interval_min, interval_max);
        random_num = a(e);
    }
    else if (interval_max > 0)//min<0&&max>0 的情况
    {
        uniform_int_distribution<unsigned> a(0, -interval_min);
        random_num = a(e);
        random_num = -random_num;
        uniform_int_distribution<unsigned> b(0, 10);
        if (b(e) % 2 == 0)
        {
            uniform_int_distribution<unsigned> c(0, interval_max);
            random_num = c(e);
        }
    }
    else//max<0的情况
    {
        uniform_int_distribution<unsigned> a(-interval_max, -interval_min);
        random_num = a(e);
        random_num = -random_num;
    }
    return random_num;

}

// { PUT, fields_vec[1], fields_vec[secondary_field_id], fields_vec[hermit_field_id] }
void input_data() {
  string field_1, field_2, field_3;
  vector<vector<string> > table;
  ifstream fin;
  fin.open("data.csv", ios::in);
    vector<string> item;				//用于存放文件中的一行数据
    string temp;						//把文件中的一行数据作为字符串放入容器中
    while (getline(fin, temp))          //利用getline（）读取每一行，并放入到 item 中
    {
        item.push_back(temp);
    }

    for (auto it = item.begin(); it != item.end(); it++)
    {
        string item1, item2, item3;
        istringstream istr(*it);                 //其作用是把字符串分解为单词(在此处就是把一行数据分为单个数据)
        string str;
        int count = 1;							 //统计一行数据中单个数据个数
        //获取文件中的第 1、2 列数据
        while (istr >> str)                      //以空格为界，把istringstream中数据取出放入到依次s中
        {
            //获取第1列数据
            if (count == 1)
            {
                item1 = str.c_str();
                colA.push_back(item1);
            }
                //获取第2列数据
            else if (count == 2)
            {
                item2 = str.c_str();
                colB.push_back(item2);
            }else if(count == 3){
                item3 = str.c_str();
                colC.push_back(item3);
            }
            count++;
        }
        table.push_back(vector<string>{ item1, item2, item3 });
    }
  hermit::build_trs(table);
  fin.close();
}

vector<string> read_data(){
    string field_1, field_2, field_3;
    ifstream fin;
    fin.open("test.csv", ios::in);
    vector<string> item;				//用于存放文件中的一行数据
    string temp;						//把文件中的一行数据作为字符串放入容器中
    while (getline(fin, temp))          //利用getline（）读取每一行，并放入到 item 中
    {
        item.push_back(temp);
    }
    fin.close();
    return item;
}

vector<pair<int,int>> prepare_pdata(){
    vector<pair<int, int>> result;
    vector<string> temp = read_data();
    for (int i = 0; i < 1000000; ++i) {
        pair<int, int> d;
        d.first = stoi(temp[i].data());
        d.second = stoi(temp[i].data());
        result.push_back(d);
    }
    return result;
}

vector<pair<int, int>> prepare_rdata(){
    vector<pair<int, int>> result;
    vector<string> temp = read_data();
    for (int i = 0; i < 1000000; ++i) {
        pair<int, int> d;
        d.first = stoi(temp[i].data());
        d.second = d.first + 0.0001 * 20000000;
        result.push_back(d);
    }
    return result;
}

vector<vector<int>> prepare_idata(){
    vector<vector<int>> result;
    vector<string> temp = read_data();
    for (int i = 0; i < 1000000; ++i) {
        vector<int> d;
        int m = atoi(temp[i].data());
        d.push_back(20000000 + i);
        int n = RandomCreatFunc(-10,10);
        d.push_back(m * 2 + n);
        d.push_back(m);
        result.push_back(d);
    }
    return result;
}

void inialDB(MWayBPTree &BT_A, MWayBPTree &BT_B,MWayBPTree &BT_A_1,MWayBPTree &BT_B_1,MWayBPTree &  BT_C_1){
    for(int i = 0;i < colA.size(); i++){
        if(i == 0){
            BT_A.init(atoi(colA[i].data()), 1);
            BT_B.init(atoi(colB[i].data()),atoi(colA[i].data()));
            BT_C_1.init(atoi(colC[i].data()),atoi(colB[i].data()));
            BT_A_1.init(atoi(colA[i].data()), 1);
            BT_B_1.init(atoi(colB[i].data()),atoi(colA[i].data()));
        }
        else{
            BT_A.insert(atoi(colA[i].data()), 1);
            BT_B.insert(atoi(colB[i].data()),atoi(colA[i].data()));
            BT_C_1.insert(atoi(colC[i].data()),atoi(colB[i].data()));
            BT_A_1.insert(atoi(colA[i].data()), 1);
            BT_B_1.insert(atoi(colB[i].data()),atoi(colA[i].data()));
        }
    }
    cout<<"初始化结束"<<endl;
}

void SearchDB(pair<int,int> range, MWayBPTree BT_A, MWayBPTree BT_B){
    auto p = hermit::lookup(to_string((range.first)), to_string((range.second)));

        // step1 hermit lookup
        set<string>& range_secondary_keys = p.first;
        set<string>& range_tuple_keys = p.second;

        //step2 host index lookup
        set<float> result_1;
        if(range_secondary_keys.size() > 0){
            result_1 = BT_B.search(atoi((*range_secondary_keys.begin()).data()), atoi((*range_secondary_keys.rbegin()).data()));
        }

        //step3 primary index lookup
        set<float> result_2;
        if(result_1.size() > 0){
            result_2 = BT_A.search(int(*result_1.begin()),int(*result_1.rbegin()));
        }

        if(range_tuple_keys.size() > 0){
            set<string>::iterator i;
            for( i=range_tuple_keys.begin();i!=range_tuple_keys.end();i++){
                BT_A.search(atoi((*i).data()));
//        cout<<*i<<" ";
            }
        }


}

void insertDB(int hmkey, int sc_key, int key, MWayBPTree &BT_A, MWayBPTree &BT_B){
    hermit::trs -> insert(hmkey, sc_key, key);
    BT_B.insert(sc_key, key);
    BT_A.insert(key, 1);
}

void CalculateMem(MWayBPTree BT_A, MWayBPTree BT_B){
    long long int sum_1 = hermit::trs ->calMem();
    long long int sum = sum_1 + BT_A.calMem() + BT_B.calMem();
    cout<<"hermit Memory:"<<to_string((sum * 4))<<endl;
}

void SearchDB_1(pair<int, int> range, MWayBPTree BT_A_1,MWayBPTree BT_B_1,MWayBPTree BT_C_1){
    set<float> range_1 = BT_C_1.search(range.first, range.second);
    if(range_1.size() > 0){
        set<float> range_2 = BT_B_1.search(int(*range_1.begin()),int(*range_1.rbegin()));
        set<float> range_3 = BT_A_1.search(int(*range_2.begin()),int(*range_2.rbegin()));
    }
}

void insertDB_1(int hmkey, int sc_key, int key, MWayBPTree &BT_A_1,MWayBPTree &BT_B_1,MWayBPTree &BT_C_1){
    BT_C_1.insert(hmkey, sc_key);
    BT_B_1.insert(sc_key, key);
    BT_A_1.insert(key, 1);
}

void CalculateMem_1(MWayBPTree BT_A_1,MWayBPTree BT_B_1,MWayBPTree BT_C_1){
    int sum = BT_A_1.calMem() + BT_B_1.calMem() + BT_C_1.calMem();
    cout<<"baseline Memory:"<<to_string((sum * 4))<<endl;
}

void createData(){
    ofstream outFile;
    outFile.open("data.csv");
    vector<long long int> temp;
    for (int i = 0; i < 20000000; ++i) {
        long long int m = rand();
        temp.push_back(m);
    }
    std::sort(temp.begin(), temp.end());
    for (int i = 0; i < 20000000; ++i) {
        int n = RandomCreatFunc(-10,10);
        outFile<<i+1<<" "<<temp[i] * 2 + n<<" "<<temp[i]<<endl;
    }
    outFile.close();

    ofstream outFile_1;
    outFile_1.open("test.csv");
    for (int i = 0; i < 1000000; ++i) {
        long long int m = rand();
        while(m>temp[temp.size() - 1]){

        }
        outFile_1<<m<<endl;
    }
    outFile_1.close();


}

/**
 * 如果测试时间太长，可以分别进行HERMIT和BASELINE的性能测试，可以注释掉不相关的代码，
 */

int main(int argc, char** args) {
    system("chcp 65001");

    MWayBPTree BT_A;
    MWayBPTree BT_B;

    MWayBPTree BT_A_1;
    MWayBPTree BT_B_1;
    MWayBPTree BT_C_1;

    input_data();
    inialDB(BT_A, BT_B, BT_A_1, BT_B_1, BT_C_1);

    vector<pair<int, int>> data_point = prepare_pdata();
    // 点查询
    clock_t start = clock();
    for (int i = 0; i < data_point.size(); ++i) {
        SearchDB(data_point[i], BT_A, BT_B);
        // 测试Baseline时用下面这个代码
        // SearchDB_1(data_range[i], BT_A_1, BT_B_1, BT_C_1);
    }
    clock_t end = clock();
    double time = double(end-start)/CLOCKS_PER_SEC;     // s
    cout<<"点查询1M条数据所花费时间："<<time<<" s"<<endl;
    // 存储空间消耗
    CalculateMem(BT_A, BT_B);
    // 测试Baseline时用下面这个代码
    // CalculateMem_1(BT_A_1, BT_B_1, BT_C_1);

    vector<pair<int, int>> data_range = prepare_rdata();
    // 范围查询
    clock_t start_1 = clock();
    for (int i = 0; i < data_range.size(); ++i) {
        SearchDB(data_range[i], BT_A, BT_B);
        // 测试Baseline时用下面这个代码
        // SearchDB_1(data_range[i], BT_A_1, BT_B_1, BT_C_1);
    }
    clock_t end_1 = clock();
    double time_1 = double(end_1-start_1)/CLOCKS_PER_SEC;     // s
    cout<<"范围查询1M条数据所花费时间："<<time_1<<" s"<<endl;
    vector<vector<int>> data_insert = prepare_idata();
     // 插入
    clock_t start_2 = clock();
    for (int i = 0; i < data_insert.size(); ++i) {
        insertDB(data_insert[i][2], data_insert[i][1], data_insert[i][0], BT_A, BT_B);
        // 测试Baseline时用下面这个代码
        // insertDB_1(data_insert[i][2], data_insert[i][1], data_insert[i][0], BT_A_1, BT_B_1, BT_C_1);
    }
    clock_t end_2 = clock();
    double time_2 = double(end_2-start_2)/CLOCKS_PER_SEC;     // s
    cout<<"插入1M条数据所花费时间："<<time_2<<" s"<<endl;

    return 0;
}

