#pragma once

#include <iostream>
#include <string>
#include <sys/types.h>
#include <cryptopp/cryptlib.h>
#include <cryptopp/sha.h>
#include <cryptopp/files.h>
#include <cryptopp/hex.h>
#include <cryptopp/filters.h>
#include <alibabacloud/oss/OssClient.h>

using namespace AlibabaCloud::OSS;

// 禁止类的拷贝、构造，防止出现访问冲突、多次delete问题，这也是单例模式（一个类只能有一个实例）的要求
// 这种写法使禁止拷贝的这部分代码可以复用，很优美
#define DISABLE_COPY_MOVE(cls)                                                                     \
    cls(const cls&) = delete;                                                                      \
    cls(cls&&) = delete;                                                                           \
    cls& operator=(const cls&) = delete;                                                          \
    cls& operator=(cls&&) = delete;

// DEFER()的作用是？？？类似golang的延迟执行？
// 因为 defer 关键字的特性和 C++ 的类的析构函数类似？？待测试功能
#define DEFER(...)                                                                                 \
    auto STDEX_NAMELNO__(_stdex_defer_, __LINE__) = stdex::make_guard([&] { __VA_ARGS__; });
#define STDEX_NAMELNO__(name, lno) STDEX_CAT__(name, lno)
#define STDEX_CAT__(a, b) a##b

namespace stdex
{

// https://zhuanlan.zhihu.com/p/21303431
template <typename Func>
struct scope_guard
{
    // explicit表示显示声明构造函数，必须显示调用（不智能调用）
    // std::move()表示把左值转换为右值，这样能更高效的使用内存
    explicit scope_guard(Func&& on_exit) : on_exit_(std::move(on_exit)) {}

    scope_guard(scope_guard const&) = delete;
    scope_guard& operator=(scope_guard const&) = delete;

    scope_guard(scope_guard&& other) : on_exit_(std::move(other.on_exit_)) {}

    ~scope_guard()
    {
        try
        {
            on_exit_();
        }
        catch (...)
        {
        }
    }

private:
    Func on_exit_;
};

// 模版函数
// std::forward<Func>(f)表示完美转发，注意区分左值和右值，简单理解右值就是常量
template <typename Func>
scope_guard<Func> make_guard(Func&& f)
{
    return scope_guard<Func>(std::forward<Func>(f));
}
}    // namespace stdex

// 计算文件的 SHA256 值
std::string CalSHA256_ByFile(std::string local_path);

// 计算云端文件的 SHA256 值（先下载到本地，再计算hash）
std::string CalSHA256_ByCloudFile(std::string bucket_name, std::string cloud_path, std::shared_ptr<OssClient> client);

// 计算数据的 SHA256 值
// 这里的byte要用CryptoPP作用域，因为std作用域中的是个类，不是unsigned char
std::string CalSHA256_ByMem(const CryptoPP::byte *data, size_t length);

// 取出路径中的文件名（目录名）
// 情况1:a/b.txt    结果：b.txt
// 情况2:a/c/       结果：c/
inline std::string GetFileName(std::string path) noexcept
{
	std::string result;
	// 如果是目录的话，先去掉最后面的/
	if (path[path.size() - 1] == '/')
	{
		path.pop_back();
		int pos = path.find_last_of('/');
		result = path.substr(pos + 1, path.size());
		result.push_back('/');
	}
	// 如果是文件
	else
	{
		int pos = path.find_last_of('/');
		result = path.substr(pos + 1, path.size());
	}

	return result;
}

// unix下的获取当前时间方法（logger用）
inline void get_current_time(timespec& current_time)  noexcept
{
	clock_gettime(CLOCK_REALTIME, &current_time);
}

inline void get_current_time_in_tm(struct tm* tm, int* nanoseconds)  noexcept
{
	timespec now;
	get_current_time(now);
	if (tm)
		gmtime_r(&now.tv_sec, tm);
	if (nanoseconds)
		*nanoseconds = static_cast<int>(now.tv_nsec);
}

// 实现类似print的功能，返回个string
std::string vstrprintf(const char* format, va_list args);

std::string strprintf(const char* format, ...);

