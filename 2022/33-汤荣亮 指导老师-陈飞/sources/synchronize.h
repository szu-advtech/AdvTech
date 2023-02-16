#pragma once

#include "cfs.h"
#include "state.h"
#include "subject.h"

class Synchronize
{
private:
    // CloudFileSystem对象
    std::shared_ptr<CloudFileSystem> cfs;

    // 观察者模式对象
    std::shared_ptr<Subject> subject;
    std::shared_ptr<Observer> observer;

    // 路径信息
    std::string config_history_path;
    std::string config_local_path;
    std::string config_cloud_path;

    // 状态树
    std::shared_ptr<StateBase> metatree_cloud;
    std::shared_ptr<StateBase> metatree_cloud_history;
    std::shared_ptr<StateBase> metatree_local;
    std::shared_ptr<StateBase> metatree_local_history;

public:
    // 构造函数
    Synchronize(std::shared_ptr<CloudFileSystem> cfs);
    // 析构函数
    ~Synchronize();

    // 初始化云同步系统
    void initialize();
    // push算法
    void algorithm_push(std::shared_ptr<StateBase> local, std::shared_ptr<StateBase> local_history, std::string local_path, std::string cloud_path, std::shared_ptr<StateBase> cloud_history);
    // pull算法
    void algorithm_pull(std::shared_ptr<StateBase> cloud, std::shared_ptr<StateBase> cloud_history, std::string cloud_path, std::string local_path, std::shared_ptr<StateBase> local_history);
    // 保存本地和云端历史树到磁盘
    void save_history();
    // 进行一轮同步函数
    void synchronize();
    // 云同步系统启动函数
    void start();
};









