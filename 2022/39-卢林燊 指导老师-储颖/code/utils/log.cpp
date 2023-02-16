//
// Created by lj on 2022/11/12.
//

#include "log.h"

#include "iostream"
#include "log.h"
#include "ctime"

bool Log::logDebug = false;
bool Log::logInfo = false;
bool Log::logWarn = false;

void Log::setLog(bool debug, bool info, bool warn) {
    Log::logDebug = debug;
    Log::logInfo = info;
    Log::logWarn = warn;
}

void Log::debug(const std::string &msg) {
    if(logDebug){
        std::cout << "[debug]" << getCurrentTimeStr() << " " << msg << std::endl;
    }
}

void Log::info(const std::string &msg) {
    if(logInfo){
        std::cout << "[info]" << getCurrentTimeStr() << " " << msg << std::endl;
    }
}

void Log::warn(const std::string &msg) {
    if(logDebug){
        std::cout << "[warn]" << getCurrentTimeStr() << " " << msg << std::endl;
    }
}

struct tm Log::getCurrentTime() {
    // 指向当前时间
    time_t nowTime;
    struct tm tmInfo;
    time(&nowTime);
    // 将当前时间本地化，并保存在tmInfo中
    localtime_s(&tmInfo, &nowTime);
    return tmInfo;
}

std::string Log::getCurrentTimeStr() {
    const tm &tm = getCurrentTime();
    return std::to_string(tm.tm_year+1900) + "-" + std::to_string(tm.tm_mon+1) +
           "-" + std::to_string(tm.tm_mday) + " " + std::to_string(tm.tm_hour) +
           ":" + std::to_string(tm.tm_min) + ":" + std::to_string(tm.tm_sec);
}