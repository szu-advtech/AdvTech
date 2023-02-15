#pragma once

#include "cfs.h"

class Subject;

class Observer
{
private:
    std::shared_ptr<CloudFileSystem> cfs;

public:
    Observer(std::shared_ptr<CloudFileSystem> incfs) : cfs(incfs) {}
    ~Observer(){}
    // 上传本地目录到云端（递归）
    void create_cloud_folder(const std::string &local_path, const std::string &cloud_path);
    // 下载云端目录到本地（递归）
    void create_local_folder(const std::string &cloud_path, const std::string &local_path);
    // 从本地上传文件到云端
    void upload(const std::string &local_path, const std::string &cloud_path);
    // 从云端下载文件到本地
    void download(const std::string &cloud_path, const std::string &local_path);
    // 删除云端文件
    void delete_cloud_file(const std::string &cloud_path);
    // 删除本地文件
    void delete_local_file(const std::string &local_path);
    // 删除云端目录
    void delete_cloud_folder(const std::string &cloud_path);
    // 删除本地目录
    void delete_local_folder(const std::string &local_path);
    // 更新云端文件
    void update_cloud_file(const std::string &local_path, const std::string &cloud_path);
    // 更新本地文件
    void update_local_file(const std::string &cloud_path, const std::string &local_path);
    // 重命名云端文件
    void rename_cloud_file(const std::string &old_cloud_path, const std::string &new_cloud_path);
    // 重命名本地文件
    void rename_local_file(const std::string &old_local_path, const std::string &new_local_path);
    // 重命名云端文件夹
    void rename_cloud_folder(const std::string &old_cloud_path, const std::string &new_cloud_path);
    // 重命名本地文件夹
    void rename_local_folder(const std::string &old_local_path, const std::string &new_local_path);

    // 更新操作：被观察者会通知观察者来执行update函数
    // 这个函数会根据被观察者的task_index和from_path、to_path来执行不同的操作方法
    void Update(Subject *subject);
    
};