#include "myutils.h"

#include <filesystem>

// 计算文件的 SHA256 值
std::string CalSHA256_ByFile(std::string local_path)
{
	std::string value;
	CryptoPP::SHA256 sha256;
	CryptoPP::FileSource(local_path.c_str(), true, new CryptoPP::HashFilter(sha256, new CryptoPP::HexEncoder(new CryptoPP::StringSink(value))));
	return value;
}

std::string temp_file_path = std::filesystem::current_path().string() + "/cloudsync_temp_file";
// 计算云端文件的 SHA256 值（先下载到本地，再计算hash）
std::string CalSHA256_ByCloudFile(std::string bucket_name, std::string cloud_path, std::shared_ptr<OssClient> client)
{
	GetObjectRequest request(bucket_name, cloud_path);
	request.setResponseStreamFactory([=]()
									 { return std::make_shared<std::fstream>(temp_file_path, std::ios_base::out | std::ios_base::in | std::ios_base::trunc | std::ios_base::binary); });

	auto outcome = client->GetObject(request);

	std::string value;
	CryptoPP::SHA256 sha256;
	CryptoPP::FileSource(temp_file_path.c_str(), true, new CryptoPP::HashFilter(sha256, new CryptoPP::HexEncoder(new CryptoPP::StringSink(value))));

    std::filesystem::remove(temp_file_path);
	return value;
}

// 计算数据的 SHA256 值
// 这里的byte要用CryptoPP作用域，因为std作用域中的是个类，不是unsigned char
std::string CalSHA256_ByMem(const CryptoPP::byte *data, size_t length)
{
	std::string value;
	CryptoPP::SHA256 sha256;
	CryptoPP::StringSource(data, length, true, new CryptoPP::HashFilter(sha256, new CryptoPP::HexEncoder(new CryptoPP::StringSink(value))));
	return value;
}

// 实现类似print的功能，返回个string
std::string vstrprintf(const char* format, va_list args)
{
    va_list copied_args;
    va_copy(copied_args, args);
    const int MAX_SIZE = 4000;
    char buffer[MAX_SIZE + 1];
    // vsnprintf的作用是将参数列表按照format格式填进去，生产一个字符串存到buffer中，返回size为字符串的长度
    int size = vsnprintf(buffer, sizeof(buffer), format, copied_args);
    va_end(copied_args);
    // if (size < 0)
    //     THROW_POSIX_EXCEPTION(errno, "vsnprintf");
    if (size <= MAX_SIZE)
        return std::string(buffer, size);
    std::string result(static_cast<std::string::size_type>(size), '\0');
    vsnprintf(&result[0], size + 1, format, args);
    return result;
}

std::string strprintf(const char* format, ...)
{
    // va_list是预先定义好的一个类型，用来接受函数传入的参数，创建一个object供其他几个函数调用
    va_list args;
    // 获取到除第一个后的参数
    va_start(args, format);
    // 不理解DEFER的作用，左值转右值，提高内存利用效率？
    // va_end()的作用是释放指针，将输入的args置为 NULL？？应该不是
    DEFER(va_end(args));
    return vstrprintf(format, args);
}