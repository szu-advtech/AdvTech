//
// Created by lj on 2022/11/12.
//

#ifndef FUXIAN_LOG_H
#define FUXIAN_LOG_H

#include "string"

class Log{
private:
    static bool logDebug;
    static bool logInfo;
    static bool logWarn;
public:
    static void setLog(bool debug=false, bool info=false, bool warn=false);
    static void debug(const std::string& msg);
    static void info(const std::string& msg);
    static void warn(const std::string& msg);
    static struct tm getCurrentTime();
    static std::string getCurrentTimeStr();
};



#endif //FUXIAN_LOG_H
