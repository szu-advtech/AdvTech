#pragma once

#include "myutils.h"

#include <cerrno>
#include <exception>
#include <string>
#include <string.h>

class ExceptionBase : public std::exception
{
private:
    mutable std::string m_cached_msg;
    // Mutable fields are not thread safe in `const` functions.
    // But who accesses exception objects concurrently anyway?

protected:
    ExceptionBase();

public:
    ~ExceptionBase() override;
    virtual std::string message() const = 0;
    const char *what() const noexcept override;
};

// 在云端操作失败异常，继承ExceptionBase
class CloudOperationException final : public ExceptionBase
{
private:
    const char *m_func; // 函数名
    const char *m_file; // 文件名
    int m_line;         // 行号
    std::string m_msg;  //操作失败提示

public:
    explicit CloudOperationException(const char *func, const char *file, int line, std::string msg)
        : m_func(func), m_file(file), m_line(line), m_msg(std::move(msg))
    {
    }
    ~CloudOperationException() override;

    std::string message() const override;
};