//
// Created by 陈贤 on 2022/3/9.
//

#ifndef FPIM_TOOL_H
#define FPIM_TOOL_H

#include <numeric>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

long UpperAlign(long num, int chunk);

long LowerAlign(long num, int chunk);

std::vector<std::string> SplitString(const std::string& s, const std::string& delimiter);

int str2int(const std::string &str);

std::vector<std::vector<int>> GetCSR(const std::string &path);

std::vector<int> SeqChunking(std::vector<double> arr, int num_chunk);

template <class T>
bool IsTheSame(const std::vector<T>& arr) {
    if(arr.size() <= 1) {
        return true;
    }
    for(int i = 1; i < arr.size(); i++) {
        if (arr[i] != arr[0]) {
            return false;
        }
    }
    return true;
}


// descending order
template <typename T>
std::vector<size_t> SortIdx(const std::vector<T> &v, bool ascending = true) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    if (ascending) {
        stable_sort(idx.begin(), idx.end(),
                    [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    } else {
        stable_sort(idx.begin(), idx.end(),
                    [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    }


    return idx;
}


std::vector<int> SetIntersection(
        const std::unordered_map<int, int> &in1,
        const std::unordered_map<int, int> &in2);

std::vector<int> SetDifference(
        const std::unordered_map<int, int> &a,
        const std::unordered_map<int, int> &b);

class UnionFind {
public:

    UnionFind(int n) {
        count = n;
        pre.resize(n);
        size.resize(n, 1);
        for (int i = 0; i < n; i++) {
            pre[i] = i;
        }
    }
    int Find(int x) {
        if (pre[x] == x) {
            return x;
        }
        return pre[x] = Find(pre[x]);
    }
    bool IsConnected(int x, int y) {
        return Find(x) == Find(y);
    }
    bool IsRoot(int x) {
        return Find(x) == x;
    }
    bool Connect(int x, int y) {
        x = Find(x);
        y = Find(y);
        if (x == y) {
            return false;
        }
        if (size[x] < size[y]) {
            std::swap(x, y);
        }
        pre[y] = x;
        size[x] += size[y];
        count--;
        return true;
    }
    void Disconnect(int x) {
        pre[x] = x;
    }
private:
    std::vector<int> pre;
    std::vector<int> size;
    int count;
};

bool ErrorExit(const std::string& file, int line);

#define ASSERT(expression) (void)( \
            (!!(expression)) || \
            (ErrorExit(__FILE__, __LINE__)) \
        )

template <class T>
void Swap(T &a, T &b) {
    auto tmp = a;
    a = b;
    b = tmp;
}



#endif //FPIM_TOOL_H
