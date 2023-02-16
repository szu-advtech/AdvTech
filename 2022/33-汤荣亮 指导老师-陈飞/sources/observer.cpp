#include "subject.h"
#include "logger.h"

#include <iostream>
#include <filesystem>

// 上传本地目录到云端（递归）
void Observer::create_cloud_folder(const std::string &local_path, const std::string &cloud_path)
{
    TRACE_LOG("准备将本地文件夹%s上传至云端文件夹%s", local_path.c_str(), cloud_path.c_str());

    // 本地不存在此文件夹则warning并结束
    if (!std::filesystem::exists(local_path))
    {
        WARN_LOG("本地文件夹%s不存在，上传目录操作终止", local_path.c_str());
        return;
    }
    // 云端存在此文件夹则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(cloud_path + ".directory_tag").size() != 0)
    {
        WARN_LOG("云端文件夹%s已存在，上传目录操作终止", cloud_path.c_str());
        return;
    }

    // 开始递归上传目录中文件
    // 先创建最外层目录
    cfs->create_folder(cloud_path);
    TRACE_LOG("已创建云端文件夹%s", cloud_path.c_str());
    //遍历本地文件夹，对每个文件进行上传，对目录递归
    TRACE_LOG("开始遍历本地文件夹%s", local_path.c_str());
    for (auto&p : std::filesystem::directory_iterator(local_path))
	{   
        // 如果是目录，递归进去处理
		if (p.is_directory())
        {
            auto local = p.path().string() + "/";
            auto cloud = cloud_path + p.path().filename().string() + "/";
            TRACE_LOG("递归将本地文件夹%s上传至云端文件夹%s", local.c_str(), cloud.c_str());
            create_cloud_folder(local, cloud);
        }
        // 如果是文件，调用cfs直接上传
        else
        {
            auto local = p.path().string();
            auto cloud = cloud_path + p.path().filename().string();
            TRACE_LOG("直接将本地%s上传至云端%s", local.c_str(), cloud.c_str());
            cfs->upload(local, cloud);
        }
	}
    TRACE_LOG("完成遍历本地文件夹%s", local_path.c_str());
}

// 下载云端目录到本地（递归）
void Observer::create_local_folder(const std::string &cloud_path, const std::string &local_path)
{
    TRACE_LOG("准备将云端文件夹%s下载至本地文件夹%s", cloud_path.c_str(), local_path.c_str());

    // 云端不存在此文件夹则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(cloud_path + ".directory_tag").size() == 0)
    {
        WARN_LOG("云端文件夹%s不存在，下载目录操作终止", cloud_path.c_str());
        return;
    }
    // 本地存在此文件夹则warning并结束
    if (std::filesystem::exists(local_path))
    {
        WARN_LOG("本地文件夹%s存在，下载目录操作终止", local_path.c_str());
        return;
    }

    // 开始递归下载云端目录中文件
    // 先创建最外层目录
    std::filesystem::create_directory(local_path);
    TRACE_LOG("已创建本地文件夹%s", local_path.c_str());
    //遍历云端文件夹，对每个文件进行下载，对目录递归
    TRACE_LOG("开始遍历云端文件夹%s", cloud_path.c_str());
    for (auto&p : cfs->list_files(cloud_path))
	{   
        // 如果是目录，递归进去处理
		if (p[p.size()-1] == '/')
        {
            auto cloud = cloud_path + p;
            auto local = local_path + p;
            TRACE_LOG("递归将云端文件夹%s下载至本地文件夹%s",  cloud.c_str(), local.c_str());
            create_local_folder(cloud, local);
        }
        // 如果是文件，调用cfs直接上传
        else
        {
            auto cloud = cloud_path + p;
            auto local = local_path + p;
            TRACE_LOG("直接将云端%s下载至本地%s", cloud.c_str(), local.c_str());
            cfs->download(cloud, local);
        }
	}
    TRACE_LOG("完成遍历云端文件夹%s", cloud_path.c_str());
}

// 从本地上传文件到云端
void Observer::upload(const std::string &local_path, const std::string &cloud_path)
{
    TRACE_LOG("准备将本地%s上传至云端%s", local_path.c_str(), cloud_path.c_str());

    // 本地不存在此文件则warning并结束
    if (!std::filesystem::exists(local_path))
    {
        WARN_LOG("本地%s不存在，上传文件操作终止", local_path.c_str());
        return;
    }
    // 云端存在此文件则warning并结束
    if (cfs->stat_file(cloud_path).size() != 0)
    {
        WARN_LOG("云端%s已存在，上传文件操作终止", cloud_path.c_str());
        return;
    }

    cfs->upload(local_path, cloud_path);
    TRACE_LOG("完成将本地%s上传至云端%s", local_path.c_str(), cloud_path.c_str());
}

// 从云端下载文件到本地
void Observer::download(const std::string &cloud_path, const std::string &local_path)
{
    TRACE_LOG("准备将云端%s下载至本地%s", cloud_path.c_str(), local_path.c_str());

    // 云端不存在此文件则warning并结束
    if (cfs->stat_file(cloud_path).size() == 0)
    {
        WARN_LOG("云端%s不存在，下载文件操作终止", cloud_path.c_str());
        return;
    }
    // 本地存在此文件则warning并结束
    if (std::filesystem::exists(local_path))
    {
        WARN_LOG("本地%s已存在，下载文件操作终止", local_path.c_str());
        return;
    }

    cfs->download(cloud_path, local_path);
    TRACE_LOG("完成将云端%s下载至本地%s", cloud_path.c_str(), local_path.c_str());
}

// 删除云端文件
void Observer::delete_cloud_file(const std::string &cloud_path)
{
    TRACE_LOG("准备将云端%s删除", cloud_path.c_str());

    // 云端不存在此文件则warning并结束
    if (cfs->stat_file(cloud_path).size() == 0)
    {
        WARN_LOG("云端%s不存在，删除文件操作终止", cloud_path.c_str());
        return;
    }

    cfs->remove(cloud_path);
    TRACE_LOG("完成将云端%s删除", cloud_path.c_str());
}

// 删除本地文件
void Observer::delete_local_file(const std::string &local_path)
{
    TRACE_LOG("准备将本地%s删除", local_path.c_str());

    // 本地不存在此文件则warning并结束
    if (!std::filesystem::exists(local_path))
    {
        WARN_LOG("本地%s不存在，删除文件操作终止", local_path.c_str());
        return;
    }

    std::filesystem::remove(local_path);
    TRACE_LOG("完成将本地%s删除", local_path.c_str());
}

// 删除云端目录
void Observer::delete_cloud_folder(const std::string &cloud_path)
{
    TRACE_LOG("准备删除云端文件夹%s", cloud_path.c_str());

    // 云端不存在此文件夹则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(cloud_path + ".directory_tag").size() == 0)
    {
        WARN_LOG("云端文件夹%s不存在，下载目录操作终止", cloud_path.c_str());
        return;
    }

    // 开始递归删除云端目录中文件和子目录
    // 遍历云端文件夹，对每个文件进行下载，对目录递归
    TRACE_LOG("开始遍历云端文件夹%s", cloud_path.c_str());
    for (auto&p : cfs->list_files(cloud_path))
	{   
        // 如果是目录，递归进去处理
		if (p[p.size()-1] == '/')
        {
            auto cloud = cloud_path + p;
            TRACE_LOG("递归删除云端文件夹%s", cloud.c_str());
            delete_cloud_folder(cloud);
        }
        // 如果是文件，调用cfs直接删除
        else
        {
            auto cloud = cloud_path + p;
            TRACE_LOG("直接删除云端%s", cloud.c_str());
            cfs->remove(cloud);
        }
	}
    TRACE_LOG("完成遍历云端文件夹%s", cloud_path.c_str());
}

// 删除本地目录
void Observer::delete_local_folder(const std::string &local_path)
{
    TRACE_LOG("准备将本地文件夹%s删除", local_path.c_str());

    // 本地不存在此文件夹则warning并结束
    if (!std::filesystem::exists(local_path))
    {
        WARN_LOG("本地文件夹%s不存在，删除文件操作终止", local_path.c_str());
        return;
    }

    std::filesystem::remove_all(local_path);
    TRACE_LOG("完成将本地文件夹%s删除", local_path.c_str());
}

// 更新云端文件
void Observer::update_cloud_file(const std::string &local_path, const std::string &cloud_path)
{

    TRACE_LOG("准备将本地%s更新至云端%s", local_path.c_str(), cloud_path.c_str());

    // 本地不存在此文件则warning并结束
    if (!std::filesystem::exists(local_path))
    {
        WARN_LOG("本地%s不存在，更新云端文件操作终止", local_path.c_str());
        return;
    }

    cfs->update(local_path, cloud_path);
    TRACE_LOG("完成将本地%s更新至云端%s", local_path.c_str(), cloud_path.c_str());
}

// 更新本地文件
void Observer::update_local_file(const std::string &cloud_path, const std::string &local_path)
{
    TRACE_LOG("准备将云端%s更新至本地%s", cloud_path.c_str(), local_path.c_str());

    // 云端不存在此文件则warning并结束
    if (cfs->stat_file(cloud_path).size() == 0)
    {
        WARN_LOG("云端%s不存在，更新文件操作终止", cloud_path.c_str());
        return;
    }

    cfs->download(cloud_path, local_path);
    TRACE_LOG("完成将云端%s更新至本地%s", cloud_path.c_str(), local_path.c_str());
}

// 重命名云端文件
void Observer::rename_cloud_file(const std::string &old_cloud_path, const std::string &new_cloud_path)
{
    TRACE_LOG("准备重命名云端%s至%s", old_cloud_path.c_str(), new_cloud_path.c_str());

    // 云端不存在旧文件名则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(old_cloud_path).size() == 0)
    {
        WARN_LOG("云端%s不存在，重命名文件操作终止", old_cloud_path.c_str());
        return;
    }

    // 云端已存在新文件名则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(new_cloud_path).size() != 0)
    {
        WARN_LOG("云端%s已存在，重命名文件操作终止", new_cloud_path.c_str());
        return;
    }

    cfs->rename(old_cloud_path, new_cloud_path);
    TRACE_LOG("完成重命名云端%s至%s", old_cloud_path.c_str(), new_cloud_path.c_str());
}

// 重命名本地文件
void Observer::rename_local_file(const std::string &old_local_path, const std::string &new_local_path)
{
    TRACE_LOG("准备重命名本地%s至%s", old_local_path.c_str(), new_local_path.c_str());

    // 本地不存在旧文件名则warning并结束
    if (!std::filesystem::exists(old_local_path))
    {
        WARN_LOG("本地%s不存在，重命名文件操作终止", old_local_path.c_str());
        return;
    }

    // 本地已存在新文件名则warning并结束
    if (std::filesystem::exists(new_local_path))
    {
        WARN_LOG("本地%s已存在，重命名文件操作终止", new_local_path.c_str());
        return;
    }

    std::filesystem::rename(old_local_path, new_local_path);
    TRACE_LOG("完成重命名本地%s至%s", old_local_path.c_str(), new_local_path.c_str());
}

// 重命名云端文件夹
void Observer::rename_cloud_folder(const std::string &old_cloud_path, const std::string &new_cloud_path)
{
    TRACE_LOG("准备重命名云端文件夹%s至%s", old_cloud_path.c_str(), new_cloud_path.c_str());

    // 云端不存在旧文件名则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(old_cloud_path + ".directory_tag").size() == 0)
    {
        WARN_LOG("云端文件夹%s不存在，重命名文件操作终止", old_cloud_path.c_str());
        return;
    }

    // 云端已存在新文件名则warning并结束
    // 注意每个目录中都有.directory_tag文件，无奈之举
    if (cfs->stat_file(new_cloud_path + ".directory_tag").size() != 0)
    {
        WARN_LOG("云端文件夹%s已存在，重命名文件操作终止", new_cloud_path.c_str());
        return;
    }

    cfs->rename(old_cloud_path, new_cloud_path);
    TRACE_LOG("完成重命名云端文件夹%s至%s", old_cloud_path.c_str(), new_cloud_path.c_str());
}

// 重命名本地文件夹
void Observer::rename_local_folder(const std::string &old_local_path, const std::string &new_local_path)
{
    TRACE_LOG("准备重命名本地文件夹%s至%s", old_local_path.c_str(), new_local_path.c_str());

    // 本地不存在旧文件名则warning并结束
    if (!std::filesystem::exists(old_local_path))
    {
        WARN_LOG("本地文件夹%s不存在，重命名文件操作终止", old_local_path.c_str());
        return;
    }

    // 本地已存在新文件名则warning并结束
    if (std::filesystem::exists(new_local_path))
    {
        WARN_LOG("本地文件夹%s已存在，重命名文件操作终止", new_local_path.c_str());
        return;
    }

    std::filesystem::rename(old_local_path, new_local_path);
    TRACE_LOG("完成重命名本地文件夹%s至%s", old_local_path.c_str(), new_local_path.c_str());
}

// 更新操作：被观察者会通知观察者来执行update函数
// 这个函数会根据被观察者的task_index和from_path、to_path来执行不同的操作方法
void Observer::Update(Subject *subject)
{

    switch (subject->task)
    {
    case CREATE_CLOUD_FOLDER:
        create_cloud_folder(subject->from_path, subject->to_path);
        break;
    case CREATE_LOCAL_FOLDER:
        create_local_folder(subject->from_path, subject->to_path);
        break;
    case UPLOAD_FILE:
        upload(subject->from_path, subject->to_path);
        break;
    case DELETE_CLOUD_FILE:
        delete_cloud_file(subject->from_path);
        break;
    case DELETE_LOCAL_FILE:
        delete_local_file(subject->from_path);
        break;
    case DELETE_CLOUD_FOLDER:
        delete_cloud_folder(subject->from_path);
        break;
    case DELETE_LOCAL_FOLDER:
        delete_local_folder(subject->from_path);
        break;
    case UPDATE_CLOUD_FILE:
        update_cloud_file(subject->from_path, subject->to_path);
        break;
    case UPDATE_LOCAL_FILE:
        update_local_file(subject->from_path, subject->to_path);
        break;
    case RENAME_CLOUD_FILE:
        rename_cloud_file(subject->from_path, subject->to_path);
        break;
    case RENAME_LOCAL_FILE:
        rename_local_file(subject->from_path, subject->to_path);
        break;
    case DOWNLOAD_FILE:
        download(subject->from_path, subject->to_path);
        break;
    case RENAME_CLOUD_FOLDER:
        rename_cloud_folder(subject->from_path, subject->to_path);
        break;
    case RENAME_LOCAL_FOLDER:
        rename_local_folder(subject->from_path, subject->to_path);
        break;
    default:
        break;
    }
}
