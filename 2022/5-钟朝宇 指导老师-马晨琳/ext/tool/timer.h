//
// Created by 陈贤 on 2022/3/1.
//

#ifndef FPIM_TIMER_H
#define FPIM_TIMER_H

#include <chrono>
#include <iostream>

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


#endif //FPIM_TIMER_H
