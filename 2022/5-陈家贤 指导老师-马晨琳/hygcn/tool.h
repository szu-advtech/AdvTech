//
// Created by szu on 2021/11/15.
//

#ifndef DRAMSIM3_TOOL_H
#define DRAMSIM3_TOOL_H

#include <iostream>
#include <vector>
#include <chrono>


uint64_t UpperAlign(uint64_t num, int chunk);

uint64_t LowerAlign(uint64_t num, int chunk);

std::vector<std::string> SplitString(const std::string& s, const std::string& delimiter);

int str2int(const std::string &str);

class Timer {
   public:
    Timer() { reset(); }
    void reset() { timePoint = std::chrono::steady_clock::now(); }
    void duration(const std::string& msg) {
        auto now = std::chrono::steady_clock::now();
        auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(now - timePoint);
        std::cout << "[time] " << msg << ": " << elapse.count() << " us" << std::endl;
        reset();
    }
    double getDuration() {
        auto now = std::chrono::steady_clock::now();
        auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(now - timePoint);
        return elapse.count();
    }
    double operator-(const Timer& t) const{
        auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(timePoint - t.getTimePoint());
        return elapse.count();
    }
    std::chrono::steady_clock::time_point getTimePoint() const{
        return timePoint;
    }
    static uint64_t getTimeStamp() {
        using time_stamp = std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds>;
        time_stamp ts = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
        return ts.time_since_epoch().count();
    };


   private:
    std::chrono::steady_clock::time_point timePoint;
};


#endif  // DRAMSIM3_TOOL_H
