#include "logger.h"
#include "myutils.h"

#include <stdarg.h>
#include <stdio.h>

// 这里仅实现unix下的logger，要实现windows的话参考securefs的实现方法
// 如果不是windows系统的话，引入linux的线程库
#include <pthread.h>
// pthread_self 是posix描述的线程ID（并非内核真正的线程id），相对于进程中各个线程之间的标识号，对于这个进程内是唯一的，而不同进程中，每个线程的 pthread_self() 可能返回是一样的。
static const void* current_thread_id(void) { return (void*)(pthread_self()); }

// vlog，接收到va_list类型的args，然后输出
void Logger::vlog(
    LoggingLevel level, const char* funcsig, int lineno, const char* format, va_list args) noexcept
{
    if (!m_fp || level < this->get_level())
        return;
    // 输出前半部分log
    prelog(level, funcsig, lineno);
    // 延迟，会在函数结束前执行？？
    DEFER(postlog(level));

    // 输出args（Mounting as a Unicode normalized filesystem？？）
    // C 库函数 int vfprintf(FILE *stream, const char *format, va_list arg) 使用参数列表发送格式化输出到流 stream 中
    // stream -- 这是指向 FILE 对象的指针，该 FILE 对象标识了流。
    // format -- 这是 C 字符串，包含了要被写入到流 stream 中的文本。
    vfprintf(m_fp, format, args);
}

// log，完成输出一条log功能（将args转为va_list类型，再调用vlog函数）
void Logger::log(
    LoggingLevel level, const char* funcsig, int lineno, const char* format, ...) noexcept
{
    if (!m_fp || level < this->get_level())
        return;
    // 将const char* format后的参数放入到args中
    // https://zh.cppreference.com/w/c/variadic/va_start
    va_list args;
    va_start(args, format);
    DEFER(va_end(args));    //函数退出时执行

    vlog(level, funcsig, lineno, format, args);
}

// 构造函数，默认设置日志等级为trace
// 根据m_fp来构造ConsoleColourSetter多态的POSIXColourSetter对象
Logger::Logger(FILE* fp, bool close_on_exit)
    : m_level(kLogTrace), m_fp(fp), m_close_on_exit(close_on_exit)
{
    m_console_color = ConsoleColourSetter::create_setter(m_fp);
}

// prelog输出前半部分log
// 例：[Info] [0x1189eddc0] [2022-10-26 09:08:11.066081000 UTC] [virtual int securefs::MountCommand::execute():1327]
// Logger::prelog(logo等级, 函数信息, 行数) noexcept
void Logger::prelog(LoggingLevel level, const char* funcsig, int lineno) noexcept
{
    // 如果m_fp不存在或者level 小于 this->get_level()，则不执行后面内容
    if (!m_fp || level < this->get_level())
        return;

    // 获得当前的时间
    struct tm now;
    int now_ns = 0;
    get_current_time_in_tm(&now, &now_ns);

    // (windows才执行)锁定FILE对象，以确保一致性线程同时访问FILE对象。
    // 防止其他线程访问，导致内容出错
    //flockfile(m_fp);
    // 如果构造了ConsoleColourSetter对象
    if (m_console_color)
    {
        // 根据传入的参数level，来设置控制台输出的颜色（仅warning和error设置）
        switch (level)
        {
        case kLogWarning:
            m_console_color->use(Colour::Warning);
            break;
        case kLogError:
            m_console_color->use(Colour::Error);
            break;
        default:
            break;
        }
    }

    // C 库函数 int fprintf(FILE *stream, const char *format, ...) 发送格式化输出到流 stream 中。
    // stream -- 这是指向 FILE 对象的指针，该 FILE 对象标识了流。
    // format -- 这是 C 字符串，包含了要被写入到流 stream 中的文本。
    fprintf(m_fp,
            "[%s] [%p] [%d-%02d-%02d %02d:%02d:%02d.%09d UTC] [%s:%d]    ",
            stringify(level),
            current_thread_id(),
            now.tm_year + 1900,
            now.tm_mon + 1,
            now.tm_mday,
            now.tm_hour,
            now.tm_min,
            now.tm_sec,
            now_ns,
            funcsig,
            lineno);
}

// postlog，后处理，加入换行符并刷新
void Logger::postlog(LoggingLevel level) noexcept
{
    // 如果构造了ConsoleColourSetter对象，且level为warning和error，设置颜色为默认颜色
    if (m_console_color && (level == kLogWarning || level == kLogError))
    {
        m_console_color->use(Colour::Default);
    }

    // 在文件末尾，加入'\n'
    putc('\n', m_fp);
    // 强迫将缓冲区内的数据写回参数stream 指定的文件中。
    fflush(m_fp);
    // (windows才执行)锁定FILE对象，以确保一致性线程同时访问FILE对象。
    // 防止其他线程访问，导致内容出错
    funlockfile(m_fp);
}

// 析构函数
Logger::~Logger()
{
    // 若m_close_on_exit为true，则关闭文件流，关闭后会自动解锁？
    if (m_close_on_exit)
        fclose(m_fp);
}

// 创建标准错误流（log输出在终端上）
Logger* Logger::create_stderr_logger() { return new Logger(stderr, false); }

// 创建文件流，追加（log存在文件中）
Logger* Logger::create_file_logger(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "a");
    // if (!fp)
    //     THROW_POSIX_EXCEPTION(errno, path);
    return new Logger(fp, true);
}

// 创建标准错误流，默认是显示在用户终端上的
// 别的文件可以用extern关键来，来用此处声明的全局变量
Logger* global_logger = Logger::create_stderr_logger();
// 后面如果像保存到文件的话在这该就行，暂时先这样来完成日志
//Logger* global_logger = Logger::create_file_logger("./log.txt");
