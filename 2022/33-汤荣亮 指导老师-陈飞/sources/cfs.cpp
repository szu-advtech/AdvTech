#include "cfs.h"
#include "myutils.h"
#include "exceptions.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <alibabacloud/oss/client/RetryStrategy.h>

using json = nlohmann::json;

#define UNUSED_PARAM(x) ((void)(x))
// 设置重试策略
class UserRetryStrategy : public RetryStrategy
{
public:

    /* maxRetries表示最大重试次数，scaleFactor为重试等待时间的尺度因子。*/
    UserRetryStrategy(long maxRetries = 3, long scaleFactor = 300) :
        m_scaleFactor(scaleFactor), m_maxRetries(maxRetries)  
    {}

    /* 您可以自定义shouldRetry函数，该函数用于判断是否进行重试。*/
    bool shouldRetry(const Error & error, long attemptedRetries) const;

    /* 您可以自定义calcDelayTimeMs函数，该函数用于计算重试的延迟等待时间。*/
    long calcDelayTimeMs(const Error & error, long attemptedRetries) const;

private:
    long m_scaleFactor;
    long m_maxRetries;
};

bool UserRetryStrategy::shouldRetry(const Error & error, long attemptedRetries) const
{    
    if (attemptedRetries >= m_maxRetries)
        return false;

    long responseCode = error.Status();

    // http code
    if ((responseCode == 403 && error.Message().find("RequestTimeTooSkewed") != std::string::npos) ||
        (responseCode > 499 && responseCode < 599)) {
        return true;
    }
    else {
        switch (responseCode)
        {
        // curl error code
        // https://curl.se/libcurl/c/libcurl-errors.html
        // https://itqq.net/curl-status-code-list-details-2/
        case (ERROR_CURL_BASE + 6):  //CURLE_COULDNT_RESOLVE_HOST 不能解析主机，当网络断开时，dns解析不到ip
        case (ERROR_CURL_BASE + 7):  //CURLE_COULDNT_CONNECT
        case (ERROR_CURL_BASE + 18): //CURLE_PARTIAL_FILE
        case (ERROR_CURL_BASE + 23): //CURLE_WRITE_ERROR
        case (ERROR_CURL_BASE + 28): //CURLE_OPERATION_TIMEDOUT
        case (ERROR_CURL_BASE + 52): //CURLE_GOT_NOTHING
        case (ERROR_CURL_BASE + 55): //CURLE_SEND_ERROR
        case (ERROR_CURL_BASE + 56): //CURLE_RECV_ERROR
            return true;
        default:
            break;
        };
    }

    return false;
}

long UserRetryStrategy::calcDelayTimeMs(const Error & error, long attemptedRetries) const
{
    UNUSED_PARAM(error);
    return (1 << attemptedRetries) * m_scaleFactor;
}

// 构造函数：初始化并创建client对象
CloudFileSystem::CloudFileSystem(std::string config_path)
{
    // 读取配置文件
    std::ifstream f(config_path);
    json config = json::parse(f);
    // 初始化sdk配置参数
    config_history_path = config["history_path"];
    config_local_path = config["local_path"];
    config_cloud_path = config["cloud_path"];
    access_key_id = config["access_key_id"];
    key_secret = config["key_secret"];
    endpoint = config["endpoint"];
    bucket_name = config["bucket_name"];

    // 初始化网络等资源。
    InitializeSdk();
    // 创建操作云的client对象
    ClientConfiguration conf;

    // 设置失败请求重试次数，默认为3次。
    // 这里设置为99999，尝试当网络中断的时候永远重试
    auto defaultRetryStrategy = std::make_shared<UserRetryStrategy>(99999);
    conf.retryStrategy = defaultRetryStrategy;

    // client这个shared_ptr指向堆上的OssClient对象，是一个智能指针
    client = std::make_shared<OssClient>(endpoint, access_key_id, key_secret, conf);
}

// 析构函数：释放资源
CloudFileSystem::~CloudFileSystem()
{
    // 释放网络等资源。
    ShutdownSdk();
}

// 上传本地文件（目录）到云端
void CloudFileSystem::upload(const std::string &local_path, const std::string &cloud_path)
{
    // 如果为目录，则创建目录（注意这里不递归处理，进一步封装的时候才用到递归）
    if (local_path[local_path.size() - 1] == '/')
    {
        create_folder(cloud_path);

        return;
    }

    // 填写本地文件完整路径，例如D:\\localpath\\examplefile.txt，其中localpath为本地文件examplefile.txt所在本地路径。
    std::shared_ptr<std::iostream> content = std::make_shared<std::fstream>(local_path, std::ios::in | std::ios::binary);
    PutObjectRequest request(bucket_name, cloud_path, content);

    // 计算文件的hash
    std::string hash = CalSHA256_ByFile(local_path);

    // 生成当前的mtime
    time_t curtime = time(NULL);
    std::string mtime = std::to_string(curtime);

    /*（可选）请参见如下示例设置访问权限ACL为私有（private）以及存储类型为标准存储（Standard）。*/
    request.MetaData().addHeader("x-oss-meta-hash", hash);
    request.MetaData().addHeader("x-oss-meta-mtime", mtime);

    auto outcome = client->PutObject(request);

     /* 异常处理。*/
    // if (!outcome.isSuccess())
    // {
    //     // 抛出异常
    //     std::string tip = "PutObject fail,code:" + outcome.error().Code() + ",message:" + outcome.error().Message() + ",requestId:" + outcome.error().RequestId();
    //     throw CloudOperationException(__FUNCTION__, __FILE__, __LINE__, std::move(tip));
    // }
}

// 下载云端文件到本地
void CloudFileSystem::download(const std::string &cloud_path, const std::string &local_path)
{
    GetObjectRequest request(bucket_name, cloud_path);
    request.setResponseStreamFactory([=]()
                                     { return std::make_shared<std::fstream>(local_path, std::ios_base::out | std::ios_base::in | std::ios_base::trunc | std::ios_base::binary); });

    auto outcome = client->GetObject(request);

    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[upload]"
    //               << "GetObjectToFile fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }
}

// 删除云端文件（空目录的话删除.directory_tag文件就行）
void CloudFileSystem::remove(const std::string &cloud_path)
{
    DeleteObjectRequest request(bucket_name, cloud_path);
    auto outcome = client->DeleteObject(request);

    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[remove]"
    //               << "DeleteObject fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }
}

// 更新云端文件（hash和mtime在下一次封装判断）
void CloudFileSystem::update(const std::string local_path, const std::string cloud_path)
{
    // 如果云路径是目录或对象不存在则结束（目录更新为另一个函数，不存在说明不应该执行更新操作）
    if (local_path[local_path.size() - 1] == '/' || !client->DoesObjectExist(bucket_name, cloud_path))
    {
        return;
    }

    // 上传本地文件到云端（默认同名会覆盖）
    // 填写本地文件完整路径，例如D:\\localpath\\examplefile.txt，其中localpath为本地文件examplefile.txt所在本地路径。
    std::shared_ptr<std::iostream> content = std::make_shared<std::fstream>(local_path, std::ios::in | std::ios::binary);
    PutObjectRequest request(bucket_name, cloud_path, content);

    // 计算文件的hash
    std::string hash = CalSHA256_ByFile(local_path);

    // 生成当前的mtime
    time_t curtime = time(NULL);
    std::string mtime = std::to_string(curtime);

    /*（可选）请参见如下示例设置访问权限ACL为私有（private）以及存储类型为标准存储（Standard）。*/
    request.MetaData().addHeader("x-oss-meta-hash", hash);
    request.MetaData().addHeader("x-oss-meta-mtime", mtime);

    auto outcome = client->PutObject(request);

    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[update]"
    //               << "PutObject fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }
}

// 对云端文件进行重命名（处理目录需要用到递归）
// 利用云端copy实现，减少网络资源消耗
void CloudFileSystem::rename(std::string old_cloud_path, std::string new_cloud_path)
{
    // 如果重命名的文件名一样，则直接退出无需重命名
    if (old_cloud_path == new_cloud_path)
    {
        return;
    }
    
    // 如果是目录，递归进去处理
    if (old_cloud_path[old_cloud_path.size() - 1] == '/')
    {
        for (auto &filename : list_files(old_cloud_path))
        {
            rename(old_cloud_path + filename, new_cloud_path + filename);
        }

        // 当递归至外层空目录的时候，跳过复制，因为c++sdk创建了文件会自动创建目录
        if (old_cloud_path[old_cloud_path.size() - 1] != '/' && new_cloud_path[new_cloud_path.size() - 1] != '/')
        {
            // 重命名（拷贝文件）处理当前函数内的云端文件
            // 还未创建的目录，云端copy api会自动创建
            // 会先创建最深层的文件，没有目录的话copy api会自动创建
            CopyObjectRequest request(bucket_name, new_cloud_path);
            request.setCopySource(bucket_name, old_cloud_path);

            /* 重命名（拷贝文件）*/

            auto outcome = client->CopyObject(request);

            // if (!outcome.isSuccess())
            // {
            //     /* 异常处理。*/
            //     std::cout << "[rename1]"
            //               << "CopyObject fail"
            //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
            // }
        }
    }
    // 如果是文件，直接处理就可以
    else
    {
        CopyObjectRequest request(bucket_name, new_cloud_path);
        request.setCopySource(bucket_name, old_cloud_path);

        /* 重命名（拷贝文件）*/
        auto outcome = client->CopyObject(request);

        // if (!outcome.isSuccess())
        // {
        //     /* 异常处理。*/
        //     std::cout << "[rename2]"
        //               << "CopyObject fail"
        //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
        // }
    }

    // 当递归至外层空目录的时候，跳过删除和修改mtime属性，因为c++sdk删除文件到目录为空后，目录会自动删除
    if (old_cloud_path[old_cloud_path.size() - 1] != '/' && new_cloud_path[new_cloud_path.size() - 1] != '/')
    {
        DeleteObjectRequest request(bucket_name, old_cloud_path);
        /* 删除旧文件。*/
        auto outcome = client->DeleteObject(request);

        // if (!outcome.isSuccess())
        // {
        //     /* 异常处理。*/
        //     std::cout << "[rename3]"
        //               << "DeleteObject fail"
        //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
        // }

        // 设置新文件的mtime最近修改时间
        time_t curtime = time(NULL);
        std::string mtime = std::to_string(curtime);
        // 如果是手动上传到oss，没有state的话这里会报错
        // 在set函数了获取request为什么老是报错？？？
        set_mtime(new_cloud_path, mtime);
    }
}

// 将云端文件1复制到云端文件2
void CloudFileSystem::copy(const std::string &src_path, const std::string &dist_path)
{
    CopyObjectRequest request(bucket_name, dist_path);
    request.setCopySource(bucket_name, src_path);

    /* 拷贝文件。*/
    auto outcome = client->CopyObject(request);

    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[copy]"
    //               << "CopyObject fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }

    // 设置新文件的mtime最近修改时间
    time_t curtime = time(NULL);
    std::string mtime = std::to_string(curtime);
    set_mtime(dist_path, mtime);
}

// 出大问题了！！！不能放入元信息
// 不要慌，不用加速目录操作的话，目录的hash和mtime都不需要用到
// 暂时解决办法：创建一个隐藏文件
void CloudFileSystem::create_folder(std::string cloud_path)
{
    // 如果后面不带/，则加上
    if (cloud_path[cloud_path.size() - 1] != '/')
    {
        cloud_path.push_back('/');
    }

    std::string temp_file_path = cloud_path + ".directory_tag";

    // temp文件
    std::shared_ptr<std::iostream> content = std::make_shared<std::stringstream>();
    // 如果为空的话，下载不到，这样可能每次pull都要调用download，所以不如随便写点东西
    *content << "ALIBABA CLOUD oss sdk is not friendly to cpp!";
    PutObjectRequest request(bucket_name, temp_file_path, content);

    auto outcome = client->PutObject(request);
    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[create_folder]"
    //               << "PutObject fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }
}

// 列出某个云端目录下的所有文件(目录)
vector<std::string> CloudFileSystem::list_files(const std::string &cloud_path)
{
    std::vector<std::string> result;
    std::string nextMarker = "";
    bool isTruncated = false;
    do
    {
        /* 列举文件。*/
        ListObjectsRequest request(bucket_name);
        /* 设置正斜线（/）为文件夹的分隔符 */
        request.setDelimiter("/");
        request.setPrefix(cloud_path);
        request.setMarker(nextMarker);
        auto outcome = client->ListObjects(request);

        if (!outcome.isSuccess())
        {
            /* 异常处理。*/
            // std::cout << "[list_files]"
            //           << "ListObjects fail"
            //           << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
            break;
        }

        for (const auto &object : outcome.result().ObjectSummarys())
        {
            if (object.Key() != cloud_path)
            {
                result.push_back(GetFileName(object.Key()));
            }
        }
        for (const auto &commonPrefix : outcome.result().CommonPrefixes())
        {
            result.push_back(GetFileName(commonPrefix));
        }
        nextMarker = outcome.result().NextMarker();
        isTruncated = outcome.result().IsTruncated();
    } while (isTruncated);

    return result;
}

// 返回云端文件的stat（目录暂时无stat）
// 如果云文件的stat中有属性是在这个函数里面生成的，需要把其设置给对应的云文件
// 因为有时候我们可能在cloud直接上传了数据，而没有通过local来操作，会出现确实云端文件缺少元信息的情况
std::map<std::string, std::string> CloudFileSystem::stat_file(const std::string &cloud_path)
{
    std::map<std::string, std::string> result;
    // 是否在这个函数内生成了元信息
    bool set_state_flag = false;

    /* 获取文件的全部元信息。*/
    auto outcome = client->HeadObject(bucket_name, cloud_path);

    if (outcome.isSuccess())
    {
        auto headMeta = outcome.result();

        // 如果不存在对应的元信息则即时生成
        if (headMeta.UserMetaData().find("hash") == headMeta.UserMetaData().end())
        {
            // 计算云端文件的hash
            result["hash"] = CalSHA256_ByCloudFile(bucket_name, cloud_path, client);

            set_state_flag = true;
        }
        else
        {
            result["hash"] = headMeta.UserMetaData()["hash"];
        }

        if (headMeta.UserMetaData().find("hash") == headMeta.UserMetaData().end())
        {
            // 生成云端文件的mtime
            time_t curtime = time(NULL);
            std::string mtime = std::to_string(curtime);
            result["mtime"] = mtime;
            set_state_flag = true;
        }
        else
        {
            result["mtime"] = headMeta.UserMetaData()["mtime"];
        }

        // 如果有即时生成的元信息，则需要写入到文件里
        if (set_state_flag)
        {
            set_stat(cloud_path, result);
        }
    }
    
    return result;
}

// 设置云端文件(目录)的stat
void CloudFileSystem::set_stat(const std::string &cloud_path, std::map<std::string, std::string> &state)
{
    std::string hash = state["hash"];
    std::string mtime = state["mtime"];

    auto meta = ObjectMetaData();

    meta.addHeader("x-oss-meta-hash", hash);
    meta.addHeader("x-oss-meta-mtime", mtime);

    CopyObjectRequest request(bucket_name, cloud_path, meta);
    request.setCopySource(bucket_name, cloud_path);

    /* 拷贝文件。*/
    auto outcome = client->CopyObject(request);

    // if (!outcome.isSuccess())
    // {
    //     /* 异常处理。*/
    //     std::cout << "[set_stat]"
    //               << "CopyObject fail"
    //               << ",code:" << outcome.error().Code() << ",message:" << outcome.error().Message() << ",requestId:" << outcome.error().RequestId() << std::endl;
    // }
}

// 设置云端文件(目录)的hash
void CloudFileSystem::set_hash(const std::string &cloud_path, const std::string &hash)
{
    std::map<std::string, std::string> state = stat_file(cloud_path);
    state["hash"] = hash;
    set_stat(cloud_path, state);
}

// 设置云端文件(目录)的mtime
void CloudFileSystem::set_mtime(const std::string &cloud_path, const std::string &mtime)
{
    std::map<std::string, std::string> state = stat_file(cloud_path);
    state["mtime"] = mtime;
    set_stat(cloud_path, state);
}