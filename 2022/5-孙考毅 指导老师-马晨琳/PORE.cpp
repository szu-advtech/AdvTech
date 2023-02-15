// PORE.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <string.h>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <iterator>
#include <io.h>
using namespace std;

typedef long long ll;
#define MAX_INT 0x3f3f3f3f

vector<string> trace_name;
vector<double> avg_respond_time;
vector<int> e_count;
vector<double> tails;
vector<double> read_hitRate;
vector<double> averageGGCC;

/** results **/
double IO_latency = 0;
double RUN_TIME = 0;
double GC_RUN_TIME = 0;
double tail_latency = 0;
double max_GC_latency = 0;
ll tl_pos = 0;
ll max_GC_pos = 0;
int IO_count = 0;
int read_count = 0;
int read_hit = 0;
int write_count = 0;
int erase_count = 0;
int GC_count = 0;
int RMW_band_count = 0;
int RMW_each_GC = 0;
int total_valid_pages_copy = 0;

vector<ll> gcpos;
vector<double> gclatency;
vector<int> gcRMW;
bool gcflag = false;

/** configurations **/
#define platter_num 16
#define band_per_platter 256
#define track_per_band 20
#define block_per_track 512
const ll block_per_band = track_per_band * block_per_track;
#define WOR_k 3         //Write head width Over Read head width

// SMR run time (ms)
#define max_seekTime 7
#define rotate_speed 7200
const double max_rotaTime = (double)60000 / rotate_speed;

const double per_band_seekTime = (double)max_seekTime / band_per_platter;
const double per_track_seekTime = per_band_seekTime / (track_per_band + WOR_k - 1);
const double per_rotaTime = (double)max_rotaTime / block_per_track;//it is also the data transmit time

const double rw_per_band_time = per_track_seekTime * (track_per_band - 1) + per_rotaTime * block_per_band;

// SSD run time (ms)
const double eraseTime = 1.5;
const double writeTime = 1.3;
const double readTime = 0.2;

#define LBA_SIZE 4096
#define PBA_SIZE 4096
#define LBAS_IN_PBA (PBA_SIZE / LBA_SIZE)
#define lba_to_pba(lba) (lba / LBAS_IN_PBA)
#define pba_to_lba(pba) (pba * LBAS_IN_PBA)

// native region
const ll data_band_num = band_per_platter;
const ll data_track_num = data_band_num * track_per_band;
const ll data_block_num = data_track_num * block_per_track;

/* SSD PC region */
#define FreeP    0				// 0 - Free page
#define ValidP   1				// 1 - Valid page
#define InvalidP 2				// 2 - Invalid page

#define page_per_block 32

struct cache_block {
    ll block_id;
    int valid_page_count;
    int free_page_count;
    int erase_count;
    int pages[page_per_block];
    int map[page_per_block];
};

const ll flash_block_num = data_block_num / page_per_block / 100;   //1% of native region
const ll reserved_block_num = flash_block_num / 10;              //10% of flash blocks
const ll PC_block_num = flash_block_num + reserved_block_num;
//const ll PC_block_num = flash_block_num;
const ll PC_page_num = PC_block_num * page_per_block;

cache_block cache_blocks[PC_block_num];
vector<ll> flash_blocks;
vector<ll> flash_blocks_pool;
vector<ll> reserved_blocks_pool;

// system ctx
ll cur_head_pos = 0;
ll cur_ppn = -1;
ll WaterMark = 0;

int open_region_size = 100;

unordered_map<ll, ll> pba_map;

void Append(cache_block& cb, ll ppn, ll pba);
void allocate_new_block(bool from_reserved);
void valid2invalid_if_exist(ll pba);

void move_op(ll pba)
{
    ll band_id, track_id, block_id;
    ll cur_band_id, cur_track_id, cur_block_id;
    band_id = pba / block_per_band;
    track_id = pba % block_per_band / block_per_track;
    block_id = pba % block_per_track;
    cur_band_id = cur_head_pos / block_per_band;
    cur_track_id = cur_head_pos % block_per_band / block_per_track;
    cur_block_id = cur_head_pos % block_per_track;

    ll seekBandCount = 0, seekTrackCount = 0, rotaCount = 0;
    seekBandCount = abs(cur_band_id - band_id);
    seekTrackCount = abs(cur_track_id - track_id);
    rotaCount = block_id - cur_block_id < 0 ? (block_id - cur_block_id + block_per_track) : block_id - cur_block_id;
    IO_latency += seekBandCount * per_band_seekTime + seekTrackCount * per_track_seekTime + rotaCount * per_rotaTime;

    cur_head_pos = pba;
}

void rw_a_block(ll pba)
{
    cur_head_pos = (pba % block_per_track + 1) % block_per_track + pba / block_per_track * block_per_track;
    IO_latency += per_rotaTime;
}

void read_cache_page(ll ppn)
{
    ll pbn = ppn / page_per_block;
    int off = ppn % page_per_block;
    assert(cache_blocks[pbn].pages[off] == ValidP);

    IO_latency += readTime;
}

ll cache_lookup(ll pba)
{
    unordered_map<ll, ll>::iterator iter = pba_map.find(pba);
    if (iter == pba_map.end())
        return -1;  //NOT_IN_CACHE
    else
        return iter->second;    //return ppn
}

void read_op(ll pba, bool in_cache)
{
    if (!in_cache)
    {
        move_op(pba);
        rw_a_block(pba);
    }
    else
    {
        read_cache_page(pba);
    }
}

void do_read(ll lba)
{
    ll pba = lba_to_pba(lba);

    ll pba_c = cache_lookup(pba);
    if (pba_c != -1){
        read_op(pba_c, true);
        ++read_hit;
    }

    else
        read_op(pba, false);

    ++read_count;
}

void unmap_assoc(cache_block& cb, int off)
{
    assert(cb.map[off] != -1);
    cb.map[off] = -1;
}

int cal_associated(unordered_set<int>& assoc)
{
    int len = flash_blocks.size();
    int pba;
    for (int i = 0; i < len; ++i)
    {
        for (int j = 0; j < page_per_block; ++j)
        {
            pba = cache_blocks[flash_blocks[i]].map[j];
            if (pba >= 0)
                assoc.insert(pba / block_per_band);
        }
    }
    return assoc.size();
}

void do_modify(ll pba)
{
    for (int i = 0; i < block_per_band; ++i)
    {
        ll pba_c = cache_lookup(pba + i);
        if (pba_c != -1)
        {
            read_op(pba_c, true);
            valid2invalid_if_exist(pba + i);
        }
    }
}

void do_rmw_band(int b)
{
    ++RMW_band_count;
    ++RMW_each_GC;

    ll pba = b * block_per_band;

    // read a band
    move_op(pba);
    IO_latency += rw_per_band_time;
    cur_head_pos = pba + (track_per_band - 1) * block_per_track;

    // modify
    do_modify(pba);

    // write a band
    move_op(pba);
    IO_latency += rw_per_band_time;
    cur_head_pos = pba + (track_per_band - 1) * block_per_track;
}

void unmap_pba_range(int begin, int end)
{
    assert(begin <= end);
    for (int i = begin; i < end; ++i)
        pba_map.erase(i);
}

void do_partialGC_every_band(int b)
{
    do_rmw_band(b);
    unmap_pba_range(b * block_per_band, (b + 1) * block_per_band);
}

void reset_cache_block(int pbn)
{
    cache_block& cb = cache_blocks[pbn];
    cb.valid_page_count = 0;
    cb.free_page_count = page_per_block;
    memset(cb.pages, FreeP, page_per_block * sizeof(int));
    memset(cb.map, -1, page_per_block * sizeof(int));
    ++cb.erase_count;

    IO_latency += eraseTime;
    ++erase_count;
}

void full_pc_clean(unordered_set<int>& assoc)
{
    for (auto it = assoc.begin(); it != assoc.end(); it++)
    {
        assert((*it) >= 0);
        do_partialGC_every_band((*it));
    }
    int len = flash_blocks.size();
    for (int i = 0; i < len; ++i)
    {
        reset_cache_block(flash_blocks[i]);
        flash_blocks_pool.push_back(flash_blocks[i]);
    }
}

void get_open_region_set(unordered_set<int>& assoc, int assoc_size)
{
    int forbid_size = assoc_size - open_region_size;
    unordered_set<int> random_pos;
    for (int i = 0; i < forbid_size; ++i)
    {
        int temp = rand() % assoc_size;
        while (random_pos.count(temp) == 1)
            temp = rand() % assoc_size;
        random_pos.insert(temp);
    }

    int t = 0, c = 0;
    for (auto it = assoc.begin(); it != assoc.end(); t++)
    {
        if (random_pos.count(t) == 1)
        {
            it = assoc.erase(it);
            random_pos.erase(t);
        }
        else
            it++;
    }
}

void pore_pc_clean(unordered_set<int>& assoc)
{
    /** open region RMW**/
    for (auto it = assoc.begin(); it != assoc.end(); it++)
    {
        assert((*it) >= 0);
        do_partialGC_every_band((*it));
    }

    /** forbidded valid page copy**/
    // copy valid pages
    int temp_blocks_count = 0;
    int i, j;
    auto it = flash_blocks.begin();
    for (i = 0; i < flash_block_num; ++i)
    {
        cache_block& cb = cache_blocks[(*it)];
        for (j = 0; j < page_per_block; ++j)
        {
            if (cb.pages[j] == ValidP)
            {
                assert(assoc.count(cb.map[j] / block_per_band) != 1);
                if (cache_blocks[cur_ppn / page_per_block].free_page_count == 0)
                {
                    allocate_new_block(true);
                    ++temp_blocks_count;
                }
                /*valid page copy*/
                IO_latency += readTime;
                Append(cache_blocks[cur_ppn / page_per_block], cur_ppn, cb.map[j]);
                cb.pages[j] = InvalidP;
                cb.map[j] = -1;

                ++total_valid_pages_copy;
            }
        }

        /*if (flash_blocks.size() == 20)
            cout << "";*/
        reset_cache_block((*it));
        if (temp_blocks_count > 0)
        {
            reserved_blocks_pool.push_back((*it));
            --temp_blocks_count;
        }
        else
            flash_blocks_pool.push_back((*it));
        it = flash_blocks.erase(it);
    }
}

void do_ssd_gc()
{
    gcflag = true;
    ++GC_count;
    RMW_each_GC = 0;

    unordered_set<int> assoc;
    int assoc_size = cal_associated(assoc);
    if (assoc_size <= open_region_size)
    {
        full_pc_clean(assoc);
        allocate_new_block(false);
    }
    else
    {
        get_open_region_set(assoc, assoc_size);
        pore_pc_clean(assoc);
        if (cache_blocks[cur_ppn / page_per_block].free_page_count == 0)
            allocate_new_block(false);
    }
}

bool get_pc_used_percentage()
{
    assert(flash_blocks_pool.size() >= 0);
    if (flash_blocks_pool.size() <= WaterMark)
        return true;
    else
        return false;
}

void allocate_new_block(bool from_reserved)
{
    ll free_id = 0, pbn;
    int min_erase = MAX_INT;
    if (from_reserved)
    {
        for (unsigned i = 0; i < reserved_blocks_pool.size(); ++i)
        {
            if (cache_blocks[reserved_blocks_pool[i]].erase_count < min_erase)
            {
                free_id = i;
                min_erase = cache_blocks[reserved_blocks_pool[i]].erase_count;
            }
        }
        pbn = reserved_blocks_pool[free_id];
        reserved_blocks_pool.erase(reserved_blocks_pool.begin() + free_id);
    }
    else
    {
        for (unsigned i = 0; i < flash_blocks_pool.size(); ++i)
        {
            if (cache_blocks[flash_blocks_pool[i]].erase_count < min_erase)
            {
                free_id = i;
                min_erase = cache_blocks[flash_blocks_pool[i]].erase_count;
            }
        }
        pbn = flash_blocks_pool[free_id];
        flash_blocks_pool.erase(flash_blocks_pool.begin() + free_id);
    }
    flash_blocks.push_back(pbn);

    cur_ppn = pbn * page_per_block;
}

int get_ppn()
{
    if (cur_ppn == -1)
        allocate_new_block(false);
    else if (cache_blocks[cur_ppn / page_per_block].free_page_count == 0)
    {
        // do_gc_if_required
        if (get_pc_used_percentage())
        {
            do_ssd_gc();    //will allocate a new block from reserved pool here
            return cur_ppn;
        }
        allocate_new_block(false);
    }
    return cur_ppn;
}

void valid2invalid_if_exist(ll pba)
{
    ll ppn = cache_lookup(pba);
    if (ppn != -1)
    {
        int off = ppn % page_per_block;
        cache_block& cb = cache_blocks[ppn / page_per_block];
        assert(cb.pages[off] == ValidP);
        cb.pages[off] = InvalidP;
        cb.valid_page_count--;
        unmap_assoc(cb, off);

        pba_map.erase(pba);
    }
}

void Append(cache_block& cb, ll ppn, ll pba)
{
    int off = ppn % page_per_block;
    assert(cb.pages[off] == FreeP);
    cb.pages[off] = ValidP;
    --cb.free_page_count;
    ++cb.valid_page_count;

    cb.map[off] = pba;
    pba_map[pba] = ppn;
    if ((cur_ppn + 1) % page_per_block != 0)
        ++cur_ppn;

    ++write_count;
    IO_latency += writeTime;
}

void do_write(ll lba)
{
    ll pba = lba_to_pba(lba);
    int ppn = get_ppn();
    cache_block& cb = cache_blocks[ppn / page_per_block];
    // do_io
    valid2invalid_if_exist(pba);
    Append(cb, ppn, pba);
}

void run_test(string inpath, string outpath)
{
    for (int i = 0; i < 1; ++i) {
        //cout<<"loop: "<<i<<endl;
        ifstream requestInput(inpath);
        stringstream ss;
        string tmp;
        int outflag = 100000;
        while (requestInput >> tmp)
        {
            gcflag = false;
            IO_latency = 0;
            ll tstamp, lba;
            string op;

            ss << tmp;
            getline(ss, tmp, ',');
            tstamp = stoll(tmp);
            getline(ss, op, ',');
            getline(ss, tmp, ',');
            lba = stoll(tmp);
            lba %= data_block_num;//映射到一个盘片上
            ss.str("");
            ss.clear();

            // debug
            /*if (IO_count == 60000)
                cout << "debug";*/

            if (op == "Read")
            {
                do_read(lba);
            }
            else if (op == "Write")
            {
                do_write(lba);
                if (IO_latency > tail_latency)
                {
                    tail_latency = IO_latency;
                    tl_pos = tstamp;
                }
            }
            else
                cout << "error." << endl;
            ++IO_count;

            RUN_TIME += IO_latency;

            if (gcflag)
            {
                GC_RUN_TIME += IO_latency;
                gcpos.push_back(tstamp);
                gclatency.push_back(IO_latency);
                gcRMW.push_back(RMW_each_GC);
                if (IO_latency > max_GC_latency)
                {
                    max_GC_latency = IO_latency;
                    max_GC_pos = tstamp;
                }
            }
            if (IO_count == outflag)
            {
                //cout << IO_count << endl;
                outflag += 100000;
            }
        }
    }
    //cout << "start new trace." << endl;
    //cout << "AVG_respond_time: " << RUN_TIME / IO_count << endl;
    //cout << "erase_count: " << erase_count << endl;
    //cout << "tail_latency: " << tail_latency << endl;
    avg_respond_time.push_back(RUN_TIME / IO_count);
    e_count.push_back(erase_count);
    tails.push_back(tail_latency);
    double sum = 0;
    for (int i = 0; i < gclatency.size(); ++i) {
        sum += gclatency[i];
    }
    averageGGCC.push_back(sum/gclatency.size());
    //read_hitRate.push_back((double)read_hit/read_count);
}

void init_sys()
{
    srand(7777);

    IO_latency = 0;
    RUN_TIME = 0;
    GC_RUN_TIME = 0;
    tail_latency = 0;
    max_GC_latency = 0;
    tl_pos = 0;
    max_GC_pos = 0;
    IO_count = 0;
    read_count = 0;
    read_hit = 0;
    write_count = 0;
    GC_count = 0;
    erase_count = 0;
    RMW_band_count = 0;
    RMW_each_GC = 0;
    total_valid_pages_copy = 0;

    cur_head_pos = 0;
    cur_ppn = -1;

    pba_map.clear();

    ll i;
    // init ssd block
    for (i = 0; i < PC_block_num; ++i)
    {
        cache_blocks[i].block_id = i;
        cache_blocks[i].erase_count = 0;
        cache_blocks[i].free_page_count = page_per_block;
        cache_blocks[i].valid_page_count = 0;
        memset(cache_blocks[i].pages, FreeP, page_per_block * sizeof(int));
        memset(cache_blocks[i].map, -1, page_per_block * sizeof(int));
    }
    // init blocks pool
    flash_blocks.clear();
    reserved_blocks_pool.clear();
    flash_blocks_pool.clear();

    for (i = 0; i < reserved_block_num; ++i)
        reserved_blocks_pool.push_back(i);
    for (; i < PC_block_num; ++i)
        flash_blocks_pool.push_back(i);

    gcpos.clear();
    gclatency.clear();
    gcRMW.clear();
    gcflag = false;
}

void getFiles(string path, vector<string> &files) {
    //文件句柄
    intptr_t hFile = 0;
    //文件信息，声明一个存储文件信息的结构体
    struct _finddata_t fileinfo;
    string p;  //字符串，存放路径
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
    {
        do {
            //如果是目录,迭代之（即文件夹内还有文件夹）
            if ((fileinfo.attrib & _A_SUBDIR)) {
                //文件名不等于"."&&文件名不等于".."
                //.表示当前目录
                //..表示当前目录的父目录
                //判断时，两者都要忽略，不然就无限递归跳不出去了！
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
            }
                //如果不是,加入列表
            else {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose函数结束查找
        _findclose(hFile);
    }
}


int main()
{
    string FilePath = "traces";//自己设置目录
    vector<string> FileName_List;
    getFiles(FilePath, FileName_List);

    for (const auto& input_path : FileName_List) {
        string filename = input_path.substr(input_path.find_last_of('\\') + 1);
        cout << filename << endl;
        trace_name.push_back(filename);
        string output_path = "output\\" + filename;

        init_sys();
        run_test(input_path, output_path);
    }

    for (int i = 0; i < trace_name.size(); ++i) {
        cout<<trace_name[i]<<","<<avg_respond_time[i]<<","<<tails[i]<<","<<averageGGCC[i]<<","<<e_count[i]<<endl;
    }
}