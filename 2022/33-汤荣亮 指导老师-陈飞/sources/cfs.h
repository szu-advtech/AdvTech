#pragma once

#include <string>
#include <alibabacloud/oss/OssClient.h>
#include <fstream>
#include <map>
#include <memory>
#include "myutils.h"

using namespace AlibabaCloud::OSS;

/* 这个类实现了操作云的基本方法，后面需要利用这些基本方法进行再封装 */

// 注意：
// 头文件中不要在类外进行定义，要定义的话在类内直接定义
// 在类外定义会引起link的时候duplicate symbol报错，原因？？？
// #paragma one和#ifndef只能保证第一次不会碰撞，所以最好每个头文件都加上
class CloudFileSystem
{
private:
    // 路径信息
    std::string config_history_path;
    std::string config_local_path;
    std::string config_cloud_path;
    // sdk key
    std::string access_key_id;
    std::string key_secret;
    std::string endpoint;
    std::string bucket_name;
    // 操作云的client对象，是一个shared_ptr
    // 后面会令其指向我们在堆上创建的OssClient对象（智能指针会自动为我们释放对象）
    std::shared_ptr<OssClient> client;

public:
    // 构造函数
    CloudFileSystem(std::string config_path);

    // 析构函数
    ~CloudFileSystem();

    // 获取成员属性
    std::string getHistoryPath()
    {
        return config_history_path;
    }
    std::string getLocalPath()
    {
        return config_local_path;
    }
    std::string getCloudPath()
    {
        return config_cloud_path;
    }

    // 不涉及递归操作
    // 上传本地文件（目录）到云端
    void upload(const std::string &local_path, const std::string &cloud_path);
    // 下载云端文件到本地
    void download(const std::string &cloud_path, const std::string &local_path);
    // 删除云端文件(目录)
    void remove(const std::string &cloud_path);
    // 更新云端文件
    void update(const std::string local_path, const std::string cloud_path);
    // 对云端文件进行重命名（处理目录需要用到递归）
    // 利用云端copy实现，减少网络资源消耗
    void rename(std::string old_cloud_path, std::string new_cloud_path);
    // 将云端文件1复制到云端文件2
    void copy(const std::string &src_path, const std::string &dist_path);
    // 创建云端文件夹
    void create_folder(std::string cloud_path);
    // 列出某个云端目录下的所有文件(目录)
    std::vector<std::string> list_files(const std::string &cloud_path);
    // 返回云端文件(目录)的stat
    std::map<std::string, std::string> stat_file(const std::string &cloud_path);
    // 设置云端文件(目录)的stat
    void set_stat(const std::string &cloud_path, std::map<std::string, std::string> &state);
    // 设置云端文件(目录)的hash
    void set_hash(const std::string &cloud_path, const std::string &hash);
    // 设置云端文件(目录)的mtime
    void set_mtime(const std::string &cloud_path, const std::string &mtime);
};
