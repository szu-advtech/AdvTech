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
using namespace std;

typedef long long ll;
#define MAX_INT 0x3f3f3f3f

// results
double IO_latency = 0;
double RUN_TIME = 0;
double BLOCK_TIME = 0;
double GC_RUN_TIME = 0;
double tail_latency = 0;
double max_GC_latency = 0;
double RW_move = 0;
ll tl_pos = 0;
ll max_GC_pos = 0;
int IO_count = 0;
int read_count = 0;
int write_count = 0;
int erase_count = 0;
int GC_count = 0;
int RMW_band_count = 0;
int RMW_each_GC = 0;
int total_valid_pages_copy = 0;

// configurations
#define platter_num 16
#define band_per_platter 1024
#define track_per_band 10
#define block_per_track 256
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
const int PC_band_num = band_per_platter / 100 + 1;     //1% of native region

const int data_band_num = (band_per_platter / (PC_band_num - 1) - 1) * (PC_band_num - 1);
//const ll data_band_num = band_per_platter;
const ll data_track_num = data_band_num * track_per_band;
const ll data_block_num = data_track_num * block_per_track;

/* PC region
const int PC_band_num = band_per_platter / 100 + 1;     //1% of native region
const int PC_track_num = PC_band_num * track_per_band;
const int PC_block_num = PC_track_num * block_per_track;*/

/* SSD PC region */
#define FreeP    0				// 0 - Free page
#define ValidP   1				// 1 - Valid page
#define InvalidP 2				// 2 - Invalid page

#define page_per_block 128
struct cache_block {
    ll block_id;
    int cur_ppn;
    int valid_page_count;
    int free_page_count;
    int erase_count;
    int pages[page_per_block];
    unordered_multiset<ll> map;  //data in pc map to native region (band-level)
};
//const ll PC_page_num = data_block_num / 100;       //1% of native region
//const ll flash_block_num = PC_page_num / page_per_block;
//const ll reserved_block_num = flash_block_num / 10;              //10% of flash blocks
//const ll PC_block_num = flash_block_num + reserved_block_num;

// super block policy
const int block_per_SBlock = block_per_band / page_per_block;
const int super_block_num = band_per_platter / 100;

const int flash_block_num = super_block_num * block_per_SBlock;
const int reserved_block_num = flash_block_num / 10;
const int PC_block_num = flash_block_num + reserved_block_num;
const int PC_page_num = flash_block_num * page_per_block;

ll WaterMark = 0;

cache_block cache_blocks[PC_block_num];
//vector<ll> flash_blocks;
//vector<ll> flash_blocks_pool;
//vector<ll> reserved_blocks_pool;

unordered_map<ll, ll> pba_map;

//const int cache_assoc = data_band_num / (PC_band_num - 1);

// system ctx
ll cur_head_pos = 0;

double IO_begin_time = 0;
double last_begin_time = 0;
double IO_end_time = 0;
double last_end_time = 0;

// unimportance
vector<ll> gcpos;
vector<double> gclatency;
vector<int> gcRMW;
bool gcflag = false;

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
    rotaCount = abs(cur_block_id - block_id);
    //IO_latency += seekBandCount * per_band_seekTime + seekTrackCount * per_track_seekTime + rotaCount * per_rotaTime; // Definition
    double move = seekBandCount * per_band_seekTime + seekTrackCount * per_track_seekTime + rotaCount * per_rotaTime;

    IO_latency += move;

    cur_head_pos = pba;

    RW_move += move;
}

void rw_a_block(ll pba)
{
    cur_head_pos = (pba % block_per_track + 1) % block_per_track + pba / block_per_track * block_per_track;
    IO_latency += per_rotaTime;
}

ll cache_lookup(ll pba)
{
    unordered_map<ll, ll>::iterator iter = pba_map.find(pba);
    if (iter == pba_map.end())
        return -1;  //NOT_IN_CACHE
    else
        return iter->second;    //return ppn
}

void read_cache_page(ll ppn)
{
    ll pbn = ppn / page_per_block;
    int off = ppn % page_per_block;
    assert(cache_blocks[pbn].pages[off] == ValidP);

    IO_latency += readTime;
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
    if (pba_c != -1)
        read_op(pba_c, true);
    else
        read_op(pba, false);

    ++read_count;
}

void unmap_assoc(cache_block& cb, int b)
{
    auto it = cb.map.find(b);
    assert(it != cb.map.end());
    cb.map.erase(it);
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
        unmap_assoc(cb, pba / block_per_band);

        pba_map[pba] = -1;
    }
}

void do_modify(ll pba)
{
    for (int i = 0; i < block_per_band; ++i)
    {
        ll pba_c = cache_lookup(pba + i);
        if (pba_c != -1)
        {
            ++total_valid_pages_copy;
            read_op(pba_c, true);
            valid2invalid_if_exist(pba + i);
        }
    }
}

void unmap_pba_range(int begin, int end)
{
    assert(begin <= end);
    for (int i = begin; i < end; ++i)
        pba_map[i] = -1;
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

void do_partialGC_every_band(int b)
{
    do_rmw_band(b);
    unmap_pba_range(b * block_per_band, (b + 1) * block_per_band);
}

// erase the victim block
void reset_cache_block(cache_block& cb)
{
    cb.cur_ppn = cb.block_id * page_per_block;
    cb.valid_page_count = 0;
    cb.free_page_count = page_per_block;
    cb.map.clear();
    memset(cb.pages, FreeP, page_per_block * sizeof(int));
    ++cb.erase_count;

    IO_latency += eraseTime;
    ++erase_count;
}

void do_ssd_gc(cache_block& cb)
{
    // move valid page data
    for (auto it = cb.map.begin(); it != cb.map.end(); )
    {
        assert((*it) >= 0);
        int b = *it;
        advance(it, cb.map.count(*it));
        do_partialGC_every_band(b);
    }

    // erase block
    reset_cache_block(cb);
}

void do_gc_if_required(int pbn)
{
    if (cache_blocks[pbn].free_page_count == 0)
    {
        gcflag = true;
        ++GC_count;
        RMW_each_GC = 0;
        do_ssd_gc(cache_blocks[pbn]);
    }
}

void Append(cache_block& cb, ll pba)
{
    int off = cb.cur_ppn % page_per_block;
    assert(cb.pages[off] == FreeP);
    cb.pages[off] = ValidP;
    --cb.free_page_count;
    ++cb.valid_page_count;

    cb.map.insert(pba / block_per_band);
    pba_map[pba] = cb.cur_ppn;
    if ((cb.cur_ppn + 1) % page_per_block != 0) //cacheBlock is not full
        ++cb.cur_ppn;

    ++write_count;
    IO_latency += writeTime;
}

void do_sblock_gc(int sbn)
{
    gcflag = true;
    ++GC_count;
    RMW_each_GC = 0;
    unordered_set<int> associated_band;
    for (int iter = sbn; iter < flash_block_num; iter += super_block_num)
    {
        cache_block& cb = cache_blocks[iter];
        for (auto it = cb.map.begin(); it != cb.map.end(); )
        {
            assert((*it) >= 0);
            int b = *it;
            advance(it, cb.map.count(*it));
            associated_band.insert(b);
        }
    }
    for (auto it = associated_band.begin(); it != associated_band.end(); it++)
    {
        do_partialGC_every_band(*it);
    }
    for (int iter = sbn; iter < flash_block_num; iter += super_block_num)
    {
        reset_cache_block(cache_blocks[iter]);
    }
}

int get_pbn(ll pba)
{
    int band_id = pba / block_per_band;
    int sbn = band_id % super_block_num;    //super block number

    for (int iter = sbn; iter < flash_block_num; iter += super_block_num)
    {
        if (cache_blocks[iter].free_page_count > 0)
            return iter;
    }

    do_sblock_gc(sbn);

    return sbn;
}

void do_write(ll lba)
{
    ll pba = lba_to_pba(lba);
    int pbn = get_pbn(pba);

    //do_gc_if_required(pbn);

    // do_io
    valid2invalid_if_exist(pba);
    Append(cache_blocks[pbn], pba);
}

void run_test(string inpath, string outpath)
{
    cout << "start new trace" << endl;
    ifstream requestInput(inpath);
    ofstream Output(outpath+".csv");
    ofstream Output_lat(outpath + "_latencies.csv");
    ofstream Output_que(outpath + "_queues.csv");
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

        /*if (IO_count == 10652)
            cout << "debug";*/

        double wait_time = 0;
        IO_begin_time = (double)tstamp / 10000; //ms
        if (IO_begin_time < last_end_time)
        {
            wait_time = last_end_time - IO_begin_time;
            Output_que << tstamp << ',' << fixed << wait_time << endl;
            IO_latency += wait_time;
            BLOCK_TIME += wait_time;
        }
        else
            Output_que << tstamp << ',' << fixed << "0" << endl;
        if (op == "Read")
            do_read(lba);
        else if (op == "Write")
            do_write(lba);
        else
            cout << "error." << endl;
        ++IO_count;
        IO_end_time = IO_begin_time + IO_latency;

        Output_lat << tstamp << ',' << fixed << IO_latency << endl;
        last_begin_time = IO_begin_time;
        last_end_time = IO_end_time;

        RUN_TIME += IO_latency;

        if (IO_latency > tail_latency)
        {
            tail_latency = IO_latency;
            tl_pos = tstamp;
        }
        if (gcflag)
        {
            GC_RUN_TIME += (IO_latency - wait_time);
            gcpos.push_back(tstamp);
            gclatency.push_back(IO_latency);
            gcRMW.push_back(RMW_each_GC);
        }
        if (IO_count == outflag)
        {
            cout << IO_count << endl;
            outflag += 100000;
        }
    }
    cout << fixed;
    cout << "BLOCK_TIME: " << BLOCK_TIME << endl;
    cout << "AVG_respond_time: " << RUN_TIME / IO_count << endl;
    cout << "AVG_cleaning time: " << GC_RUN_TIME / GC_count << endl;
    cout << "IO_count: " << IO_count << endl;
    cout << "read_count: " << read_count << endl;
    cout << "write_count: " << write_count << endl;
    cout << "GC_count: " << GC_count << endl;
    cout << "RMW_band_count: " << RMW_band_count << endl;
    cout << "erase_count: " << erase_count << endl;
    cout << "total_valid_pages_copy: " << total_valid_pages_copy << endl;
    cout << "tl_incur_pos:" << endl;
    cout << tl_pos << ' ' << tail_latency << endl;
    cout << "RW_move: " << RW_move << endl;
    /*cout << "gc_pos(timestamp, RMW_each_gc, respond_time_each_gc):" << endl;
    for (unsigned i = 0; i < gcpos.size(); ++i)
    {
        cout << gcpos[i] << ',' << gcRMW[i] << ',' << gclatency[i] << endl;
    }*/

    Output << fixed;
    Output << "Run_time: " << RUN_TIME << endl;
    Output << "BLOCK_TIME: " << BLOCK_TIME << endl;
    Output << "Cleaning time: " << GC_RUN_TIME << endl;
    Output << "AVG_respond_time: " << RUN_TIME / IO_count << endl;
    Output << "AVG_cleaning time: " << GC_RUN_TIME / GC_count << endl;
    Output << "IO_count: " << IO_count << endl;
    Output << "read_count: " << read_count << endl;
    Output << "write_count: " << write_count << endl;
    Output << "GC_count: " << GC_count << endl;
    Output << "RMW_band_count: " << RMW_band_count << endl;
    Output << "erase_count: " << erase_count << endl;
    Output << "total_valid_pages_copy: " << total_valid_pages_copy << endl;
    Output << "tl_incur_pos:" << endl;
    Output << tl_pos << ' ' << tail_latency << endl;
    Output << "RW_move: " << RW_move << endl;
    /*Output << "gc_pos:" << endl;
    for (unsigned i = 0; i < gcpos.size(); ++i)
    {
        Output << gcpos[i] << ',' << gcRMW[i] << ',' << gclatency[i] << endl;
    }*/
}

void init_sys()
{
    IO_latency = 0;
    RUN_TIME = 0;
    BLOCK_TIME = 0;
    GC_RUN_TIME = 0;
    tail_latency = 0;
    max_GC_latency = 0;
    RW_move = 0;
    tl_pos = 0;
    max_GC_pos = 0;
    IO_count = 0;
    read_count = 0;
    write_count = 0;
    GC_count = 0;
    erase_count = 0;
    RMW_band_count = 0;
    RMW_each_GC = 0;
    total_valid_pages_copy = 0;

    cur_head_pos = 0;

    pba_map.clear();

    ll i;
    // init ssd block
    for (i = 0; i < PC_block_num; ++i)
    {
        cache_blocks[i].block_id = i;
        cache_blocks[i].cur_ppn = i * page_per_block;
        cache_blocks[i].erase_count = 0;
        cache_blocks[i].free_page_count = page_per_block;
        cache_blocks[i].valid_page_count = 0;
        memset(cache_blocks[i].pages, FreeP, page_per_block * sizeof(int));

        cache_blocks[i].map.clear();
    }

    gcpos.clear();
    gclatency.clear();
    gcRMW.clear();
    gcflag = false;

    IO_begin_time = 0;
    last_begin_time = 0;
    IO_end_time = 0;
    last_end_time = 0;

    // init blocks pool
    /*flash_blocks.clear();
    reserved_blocks_pool.clear();
    flash_blocks_pool.clear();

    for (i = 0; i < reserved_block_num; ++i)
        reserved_blocks_pool.push_back(i);
    for (; i < PC_block_num; ++i)
        flash_blocks_pool.push_back(i);*/
}

int main()
{
    init_sys();
    run_test("D:/data_set/all-trace/wdev_0.csv", "D:/skylight/results/Output.csv");

/*    init_sys();
    run_test("D:/data_set/all-trace/proj_0.csv", "D:/skylight/results/out_proj_0");*/
/*    init_sys();
    run_test("D:/data_set/all-trace/mds_0.csv", "D:/skylight/results/out_mds_0");*/
/*
	init_sys();
	run_test("D:/data_set/all-trace/rsrch_0.csv", "D:/skylight/results/out_rsrch_0");
	init_sys();
	run_test("D:/data_set/all-trace/src1_2.csv", "D:/skylight/results/out_src1_2");
	init_sys();
	run_test("D:/data_set/all-trace/src2_0.csv", "D:/skylight/results/out_src2_0");
    init_sys();
	run_test("D:/data_set/all-trace/src2_2.csv", "D:/skylight/results/out_src2_2");
    init_sys();
    run_test("D:/data_set/all-trace/stg_0.csv", "D:/skylight/results/out_stg_0");
    init_sys();
    run_test("D:/data_set/all-trace/stg_1.csv", "D:/skylight/results/out_stg_1");
    init_sys();
    run_test("D:/data_set/all-trace/ts_0.csv", "D:/skylight/results/out_ts_0");
    init_sys();
    run_test("D:/data_set/all-trace/usr_0.csv", "D:/skylight/results/out_usr_0");
    init_sys();
    run_test("D:/data_set/all-trace/wdev_0.csv", "D:/skylight/results/out_wdev_0");
    init_sys();
    run_test("D:/data_set/all-trace/web_0.csv", "D:/skylight/results/out_web_0");
    init_sys();
    run_test("D:/data_set/all-trace/web_2.csv", "D:/skylight/results/out_web_2");
*/

    return 0;
}