#pragma once

#include <list>
#include <string>
#include <memory>

#include "observer.h"

// 操作枚举集合
enum OP
{
    // 创建云目录
    CREATE_CLOUD_FOLDER,
    // 创建本地目录
    CREATE_LOCAL_FOLDER,
    // 上传文件
    UPLOAD_FILE,
    // 删除云文件
    DELETE_CLOUD_FILE,
    // 删除本地文件
    DELETE_LOCAL_FILE,
    // 删除云目录
    DELETE_CLOUD_FOLDER,
    // 删除本地目录
    DELETE_LOCAL_FOLDER,
    // 更新云文件
    UPDATE_CLOUD_FILE,
    // 更新本地文件
    UPDATE_LOCAL_FILE,
    // 重命名云文件
    RENAME_CLOUD_FILE,
    // 重命名本地文件
    RENAME_LOCAL_FILE,
    // 下载文件
    DOWNLOAD_FILE,
    // 重命名云端目录
    RENAME_CLOUD_FOLDER,
    // 重命名本地目录
    RENAME_LOCAL_FOLDER
};

// 观察者模式
// 主体，更改状态后通知所有观察者
class Subject
{
public:
    std::list<std::shared_ptr<Observer>> list_observer;
    OP task;
    std::string from_path;
    std::string to_path;

public:
    Subject(){};
    ~Subject(){};
    // 添加观察者
    void Attach(std::shared_ptr<Observer> observer) noexcept;
    // 删除观察者
    void Detach(std::shared_ptr<Observer> observer) noexcept;
    // 通知观察者，观察者会做出反应
    void Notify();
    // 改变主体的状态，会通知观察者
    // 带默认值的类内方法，只能在类内定义
    void Set_data(OP task, std::string from_path = "", std::string to_path = "")
    {
        this->task = task;
        this->from_path = from_path;
        this->to_path = to_path;

        Notify();
    }
};