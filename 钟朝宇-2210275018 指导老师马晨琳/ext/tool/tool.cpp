//
// Created by 陈贤 on 2022/3/9.
//
#include <cassert>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <set>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include "tool.h"


long UpperAlign(long num, int chunk) {
    return std::ceil((double)num / chunk) * chunk;
}

long LowerAlign(long num, int chunk) {
    return std::floor((double)num / chunk) * chunk;
}

std::vector<std::string> SplitString(const std::string& s, const std::string& delimiter) {
    std::string::size_type pos1, pos2;
    std::vector<std::string> v;
    pos2 = s.find(delimiter);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + delimiter.size();
        pos2 = s.find(delimiter, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
    return v;
}

int str2int(const std::string &str) {
    return std::atoi(str.c_str());
}

std::vector<std::vector<int>> GetCSR(const std::string &path) {
    std::ifstream file(path);
    std::vector<std::vector<int>> csr;
    ASSERT(file.is_open());
    if (file.is_open()) {
        std::string line;
        int last = -1;

        while (std::getline(file, line)) {
            std::string delimiter = ",";
            std::vector<std::string> vec = SplitString(line, delimiter);
            int src = str2int(vec[0]);
            int dst = str2int(vec[1]);
            assert(last <= dst);
            last = dst;
            if(csr.size() < dst + 1) {
                csr.resize(dst + 1);
            }
            csr[dst].push_back(src);
        }
        file.close();
    }
    return csr;
}

std::vector<int> SeqChunking(std::vector<double> arr, int num_chunk){
    int n = arr.size();
    std::vector<std::vector<double>> M;
    std::vector<std::vector<int>> help;
    M.reserve(n + 1);
    help.reserve(n + 1);
    for(int i = 0; i < n + 1; i++) {
        M.emplace_back(num_chunk + 1, INT64_MAX);
        help.emplace_back(num_chunk + 1, 0);
    }
    std::vector<double> prefix(n+1, 0);
    for(int i = 0; i < n; i++) {
        prefix[i + 1] = arr[i] + prefix[i];
    }
    M[0][0] = 0;
    for(int i = 1; i <=n; i++) {
        for(int j = 1; j <= std::min(num_chunk, i); j++) {
            for(int k = 0; k < i; k++) {
                assert(i >= 0);
                assert(j >= 0);
                assert(i <= n);
                assert(j <= num_chunk);
                double what = prefix[i] - prefix[k];
                if(M[i][j] > std::max(M[k][j-1], what)) {
                    help[i][j] = k;
                }
                M[i][j] = std::min(M[i][j], std::max(M[k][j-1], what));
            }
        }
    }

    std::vector<int> pos;
    int i = n, j = num_chunk;
    while(j > 0) {
        pos.emplace_back(i);
        assert(i >= 0);
        assert(j >= 0);
        assert(i <= n);
        assert(j <= num_chunk);
        i = help[i][j];
        j -= 1;
    }
    std::reverse(pos.begin(), pos.end());
    return pos;
}


std::vector<int> SetIntersection(
        const std::unordered_map<int, int> &in1,
        const std::unordered_map<int, int> &in2) {

    if (in2.size() < in1.size()) {
        return SetIntersection(in2, in1);
    }
    std::unordered_set<int> out;
    for (auto it = in1.begin(); it != in1.end(); it++) {
        if (in2.find(it->first) != in2.end())
            out.insert(it->first);
    }
    return std::vector<int>(out.begin(), out.end());
}

std::vector<int> SetDifference(
        const std::unordered_map<int, int> &a,
        const std::unordered_map<int, int> &b) {

    std::unordered_set<int> out;
    for (auto it = a.begin(); it != a.end(); it++) {
        if (b.find(it->first) == b.end())
            out.insert(it->first);
    }
    return std::vector<int>(out.begin(), out.end());
}

bool ErrorExit(const std::string &file, int line) {
    std::cerr << "Exiting Abruptly - " << file << ":" << line << std::endl;
    std::exit(-1);
    return -1;
}
