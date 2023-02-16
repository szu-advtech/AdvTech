#include "exceptions.h"

ExceptionBase::ExceptionBase() = default;
ExceptionBase::~ExceptionBase() = default;
const char* ExceptionBase::what() const noexcept
{
    if (m_cached_msg.empty())
    {
        try
        {
            message().swap(m_cached_msg);
        }
        catch (...)
        {
            return "An exception occurred while formatting exception message";
        }
    }
    return m_cached_msg.c_str();
}

CloudOperationException::~CloudOperationException() = default;
std::string CloudOperationException::message() const
{
    return strprintf(
            "CloudOperation code executed in function \"%s\" at %s:%d.\nTip: %s", m_func, m_file, m_line, m_msg.c_str());
}
