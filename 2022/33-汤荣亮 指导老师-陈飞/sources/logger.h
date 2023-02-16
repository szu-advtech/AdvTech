#pragma once

#include "color.h"
#include "myutils.h"

#include <memory>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string>

// 日志的等级有五种，可供用户设置的为trace和verbose，默认为info级别，系统会输出>=当前日志级别的日志
// 定义枚举类型的主要目的是：增加程序的可读性。
// 枚举类型最常见也最有意义的用处之一就是用来描述状态量
enum LoggingLevel
{
    // trace，跟踪
    kLogTrace = 0, // 0为该枚举常量的序号
                   // verbose，信息
    kLogVerbose = 1,
    // info，信息
    kLogInfo = 2,
    // warning，警告
    kLogWarning = 3,
    // error，错误
    kLogError = 4
};

// 根据枚举类型返回对应的字符串
// 内联函数
// const char*表示指针所指向的值不能改，但是指针指向可以改
inline const char *stringify(LoggingLevel lvl)
{
    switch (lvl)
    {
    case kLogTrace:
        return "Trace";
    case kLogVerbose:
        return "Verbose";
    case kLogInfo:
        return "Info";
    case kLogWarning:
        return "Warning";
    case kLogError:
        return "Error";
    }
    return "UNKNOWN";
}

// 暂时不实现，递归用这种方法不好用
// 跟踪观察者中api调用类（对基础云操作方法进行再一次封装后的方法）
// class Tracer;

// 日记类，用于输出日志
class Logger
{
    // 禁止拷贝
    DISABLE_COPY_MOVE(Logger)
    // 友元，Tracer类可以直接访问Logger类里面的所有内容
    // friend class Tracer;

private:
    // log的等级
    LoggingLevel m_level;
    // 当前打开的文件流（FILE是c语言标准库中的类，用fopen()）
    FILE *m_fp;
    // ConsoleColourSetter类对象的unique_ptr，用于设置控制台输出颜色
    std::unique_ptr<ConsoleColourSetter> m_console_color;
    // 是否关闭退出log
    bool m_close_on_exit;

    // 构造函数
    explicit Logger(FILE *fp, bool close_on_exit);

    //
    void prelog(LoggingLevel level, const char *funcsig, int lineno) noexcept;
    //
    void postlog(LoggingLevel level) noexcept;

public:
    // 创建标准错误流（log输出在终端上）
    static Logger *create_stderr_logger();
    // 创建文件流，追加（log存在文件中）
    static Logger *create_file_logger(const std::string &path);

    //
    void vlog(LoggingLevel level,
              const char *funcsig,
              int lineno,
              const char *format,
              va_list args) noexcept;
    // 注意这个函数和attribute连在一起的
    // https://www.jianshu.com/p/965f6f903114
    void log(LoggingLevel level, const char *funcsig, int lineno, const char *format, ...) noexcept
// 如果不是_MSC_VER，即不是Microsoft Visual C/C++
#ifndef _MSC_VER
        __attribute__((format(printf, 5, 6)))
#endif
        ;

    // 返回当前 Logger 对象的 m_level
    LoggingLevel get_level() const noexcept { return m_level; }
    void set_level(LoggingLevel lvl) noexcept { m_level = lvl; }

    ~Logger();
};

// 想正确使用则需要先进行extern声明，意思是global_logger在其他地方进行了定义。
extern Logger *global_logger;

// windows编译器和linux获取函数名的宏定义不同，这里进行判断，然后再统一命名
#ifdef _MSC_VER
#define FULL_FUNCTION_NAME __FUNCSIG__
#else
#define FULL_FUNCTION_NAME __PRETTY_FUNCTION__
#endif

// define语法规定是要写在同一行，通过\转义符号，可以把不同行连为一行
// 用define的目的也是为了让 FULL_FUNCTION_NAME 和 __LINE__ 宏定义真正有效
// using声明使用后前面的限定空间可以略去了
// 如果global_logger->get_level() <= log_level，则调用global_logger->log()
// 宏定义重用do{}while(0)很妙：https://www.zhihu.com/question/24386599

// FULL_FUNCTION_NAME 为上面统一命名后的宏定义，返回当前函数信息
// __LINE__ 为当前源代码的行号
// __VA_ARGS__ 为获取到可变参数...
// 可变参数...表示函数参数中的最后一部分的参数
#define GENERIC_LOG(log_level, ...)                                                   \
    do                                                                                \
    {                                                                                 \
        if (global_logger && global_logger->get_level() <= log_level)                 \
        {                                                                             \
            global_logger->log(log_level, FULL_FUNCTION_NAME, __LINE__, __VA_ARGS__); \
        }                                                                             \
    } while (0)
// 定义不同级别的log函数，让调用更加直观，更加方便
// 宏定义，直接替换代码
#define TRACE_LOG(...) GENERIC_LOG(kLogTrace, __VA_ARGS__)
#define VERBOSE_LOG(...) GENERIC_LOG(kLogVerbose, __VA_ARGS__)
#define INFO_LOG(...) GENERIC_LOG(kLogInfo, __VA_ARGS__)
#define WARN_LOG(...) GENERIC_LOG(kLogWarning, __VA_ARGS__)
#define ERROR_LOG(...) GENERIC_LOG(kLogError, __VA_ARGS__)