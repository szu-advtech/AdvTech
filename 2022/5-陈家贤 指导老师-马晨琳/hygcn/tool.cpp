//
// Created by szu on 2021/11/15.
//

#include "tool.h"
#include "cmath"


uint64_t UpperAlign(uint64_t num, int chunk) {
    return std::ceil((double)num / chunk) * chunk;
}

uint64_t LowerAlign(uint64_t num, int chunk) {
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
