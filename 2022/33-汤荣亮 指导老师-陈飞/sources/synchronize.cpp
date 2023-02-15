#include "synchronize.h"
#include "logger.h"

#include <filesystem>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>

// 定义在类外（注意不能用多线程，不能多个Synchronize对象同时操作）
// 判断是否结束程序（按下ctrl+c，保证在一轮同步结束后再结束）
 bool close_tag;
// 按下ctrl+c后执行的逻辑（更改close的值）
void stop(int sig)
{
    close_tag = true;
    INFO_LOG("云同步系统关闭标志设置为True");
}

// 构造函数
Synchronize::Synchronize(std::shared_ptr<CloudFileSystem> cfs)
{
    // 初始化CloudFileSystem对象
    this->cfs = cfs;

    // 从CloudFileSystem对象中取出路径配置信息
    config_history_path = cfs->getHistoryPath();
    config_local_path = cfs->getLocalPath();
    config_cloud_path = cfs->getCloudPath();
    TRACE_LOG("历史树保存路径为%s", config_history_path.c_str());
    TRACE_LOG("本地数据目录为%s", config_local_path.c_str());
    TRACE_LOG("云端数据目录为%s", config_cloud_path.c_str());

    // 根据CloudFileSystem对象创建相应的观察者模式对象
    subject = std::make_shared<Subject>();
    observer = std::make_shared<Observer>(cfs);
    subject->Attach(observer);
    TRACE_LOG("观察者模式启动成功");

    // 初始化close为true，只是执行start函数后才是false
    close_tag = true;
    // 按下 Ctrl+C 之后停止同步程序，执行selt.stop函数
    signal(SIGINT, stop);
}

// 析构函数
Synchronize::~Synchronize() {}

// 初始化云同步系统：
// 初始化当前状态树
// 尝试从本地磁盘找历史状态树，找不到则直接创建空的历史状态树
void Synchronize::initialize()
{
    TRACE_LOG("开始初始化云同步系统");

    // 构造云端当前状态树和本地当前状态树
    // 构造云端当前树
    TRACE_LOG("构建云端当前状态树");
    metatree_cloud = generate_statetree_cloud(config_cloud_path, cfs);

    // 构造本地当前树
    TRACE_LOG("构建本地当前状态树");
    metatree_local = generate_statetree_local(config_local_path);

    // 获取云端历史状态树和本地历史状态树
    // 获取云端历史树
    // 磁盘存在云端历史树文件的话直接从文件中读取
    if (std::filesystem::exists(config_history_path + ".cloud"))
    {
        metatree_cloud_history = load_statetree(config_history_path + ".cloud");
        TRACE_LOG("在磁盘找到云端历史树文件，从磁盘获取云端历史树");
    }
    // 本地不存在云端历史树文件的话，则创建空的云端历史树对象
    else
    {
        metatree_cloud_history = std::make_shared<DirectoryState>(config_cloud_path);
        TRACE_LOG("在磁盘未找到云端历史树文件，创建云端历史树");
    }
    // 获取本地历史树
    // 磁盘存在本地历史树文件的话直接从文件中读取
    if (std::filesystem::exists(config_history_path + ".local"))
    {
        metatree_local_history = load_statetree(config_history_path + ".local");
        TRACE_LOG("在磁盘找到云端历史树文件，从磁盘获取本地历史树");
    }
    // 本地不存在云端历史树文件的话，则创建空的云端历史树对象
    else
    {
        metatree_local_history = std::make_shared<DirectoryState>(config_local_path);
        TRACE_LOG("在磁盘未找到云端历史树文件，创建本地历史树");
    }

    TRACE_LOG("完成初始化云同步系统");
}

// push算法
void Synchronize::algorithm_push(std::shared_ptr<StateBase> local, std::shared_ptr<StateBase> local_history, std::string local_path, std::string cloud_path, std::shared_ptr<StateBase> cloud_history)
{
    // 将StateBase指针转换为DirectoryState指针
    std::shared_ptr<DirectoryState> local_current_tree = std::dynamic_pointer_cast<DirectoryState>(local);
    std::shared_ptr<DirectoryState> local_history_tree = std::dynamic_pointer_cast<DirectoryState>(local_history);
    std::shared_ptr<DirectoryState> cloud_history_tree = std::dynamic_pointer_cast<DirectoryState>(cloud_history);

    TRACE_LOG("准备遍历本地当前元信息树的每一项，判断是否需要进行同步操作");
    // 处理本地新增、修改部分，上传更新到云端
    // 遍历处理当前目录下的子目录和子文件child
    for (auto &child : local_current_tree->getChildren())
    {
        // 取出child的路径
        std::string filepath = child->getFilename();
        // 对应当前child在本地的路径
        std::string next_local_path = filepath;
        // 对应child在云端的路径
        std::string next_cloud_path = cloud_path + GetFileName(filepath);
        // 在同一目录的历史树中查找child是否也存在
        // 找不到为NULL
        std::shared_ptr<StateBase> next_local_history = local_history_tree->findState(next_local_path, true, cfs, local_path, cloud_path);
        TRACE_LOG("准备处理当前项%s", filepath.c_str());
        TRACE_LOG("对应本地文件路径为%s", next_local_path.c_str());
        TRACE_LOG("对应云端文件路径为%s", next_cloud_path.c_str());
        if (next_local_history)
        {
            TRACE_LOG("本地历史树中存在");
        }
        else
        {
            TRACE_LOG("本地历史树中不存在");
        }

        // 如果当前child是目录
        if (child->getFiletype() == Directory)
        {
            TRACE_LOG("当前child为目录");
            // 如果child在本地历史树中存在，则递归push
            if (next_local_history)
            {
                std::shared_ptr<StateBase> next_cloud_history = cloud_history_tree->findState(next_cloud_path, false, cfs, local_path, cloud_path);
                
                algorithm_push(child, next_local_history, next_local_path, next_cloud_path, next_cloud_history);
            }
            // 如果child在本地历史树中不存在，则直接将该目录上传
            else
            {
                subject->Set_data(CREATE_CLOUD_FOLDER, next_local_path, next_cloud_path);
                
                // TODO
                // 这里应该递归更新历史树，历史树的更新操作跟在最底层的操作方法中，这样效率更高
                // 这样的话代码需要重构，并且为了效率，需要重新设计代码结构
                // 目前只插入目录，没有插入目录下文件，会导致push上传后再进行pull下载，虽然用if逻辑阻断了不影响效果，但是多了函数栈的消耗

                // 插入记录到本地历史树和云端历史树
                TRACE_LOG("插入记录本地历史树%s和云端历史树%s", next_local_path.c_str(), next_cloud_path.c_str());
                local_history_tree->insert(std::make_shared<DirectoryState>(next_local_path));
                cloud_history_tree->insert(std::make_shared<DirectoryState>(next_cloud_path));
            }
        }
        // 否则child就是文件
        else
        {
            TRACE_LOG("当前child为文件");
            // 如果child在本地历史树中存在，且此历史文件与本地文件摘要值不相同，且本地最新，则更新云端文件
            if (next_local_history && std::stol(child->getMtime()) > std::stol(next_local_history->getMtime()) && child->getFileid() != next_local_history->getFileid())
            {
                subject->Set_data(UPDATE_CLOUD_FILE, next_local_path, next_cloud_path);
                
                // TODO
                // 把历史树的更新跟最底层的操作写在一起，可以减少不必要操作

                // 更新记录到本地历史树和云端历史树
                TRACE_LOG("更新记录本地历史树%s和云端历史树%s", next_local_path.c_str(), next_cloud_path.c_str());
                std::shared_ptr<StateBase> temp_local = local_history_tree->findState(next_local_path, true, cfs, local_path, cloud_path);
                if (temp_local)
                {
                    temp_local->setMtime(child->getMtime());
                    temp_local->setFileid(child->getFileid());
                }
                std::shared_ptr<StateBase> temp_cloud = cloud_history_tree->findState(next_cloud_path, false, cfs, local_path, cloud_path);
                if (temp_cloud)
                {
                    std::string mtime = cfs->stat_file(next_cloud_path)["mtime"];
                    temp_cloud->setMtime(mtime);
                    temp_cloud->setFileid(child->getFileid());
                }
            }
            // 如果child在本地历史树中不存在，则直接将该文件上传
            if (!next_local_history)
            {
                subject->Set_data(UPLOAD_FILE, next_local_path, next_cloud_path);
                
                // TODO
                // 把历史树的更新跟最底层的操作写在一起，可以减少不必要操作

                // 插入记录到本地历史树和云端历史树
                TRACE_LOG("插入记录本地历史树%s和云端历史树%s", next_local_path.c_str(), next_cloud_path.c_str());
                local_history_tree->insert(std::make_shared<FileState>(next_local_path, child->getMtime(), child->getFileid()));
                std::string mtime = cfs->stat_file(next_cloud_path)["mtime"];
                cloud_history_tree->insert(std::make_shared<FileState>(next_cloud_path, mtime, child->getFileid()));
            }
        }
        TRACE_LOG("完成处理当前项%s", filepath.c_str());
    }
    TRACE_LOG("完成遍历本地当前元信息树的每一项");

    // 处理本地删除部分，在云端删除
    // 如果历史树不存在，结束运行
    if (!local_history)
    {
        return;
    }
    TRACE_LOG("准备遍历本地历史元信息树的每一项，判断是否需要进行云端文件删除操作");
    // 遍历本地历史元信息树的每一项，判断是否需要进行云端文件删除操作
    for (auto &child : local_history_tree->getChildren())
    {
        // 取出child的路径
        std::string filepath = child->getFilename();
        // 对应当前child在本地的路径
        std::string next_local_path = filepath;
        // 对应child在云端的路径
        std::string next_cloud_path = cloud_path + GetFileName(filepath);
        // 在同一目录的历史树中查找child是否也存在
        // 找不到为NULL
        std::shared_ptr<StateBase> next_local_current = local_current_tree->findState(next_local_path, true, cfs, local_path, cloud_path);
        TRACE_LOG("准备处理当前项%s", filepath.c_str());
        TRACE_LOG("对应本地文件路径为%s", next_local_path.c_str());
        TRACE_LOG("对应云端文件路径为%s", next_cloud_path.c_str());
        if (next_local_current)
        {
            TRACE_LOG("本地当前树中存在");
        }
        else
        {
            TRACE_LOG("本地当前树中不存在，在云端执行删除操作");
            if (child->getFiletype() == Directory)
            {
                subject->Set_data(DELETE_CLOUD_FOLDER, next_cloud_path);
            }
            else
            {
                subject->Set_data(DELETE_CLOUD_FILE, next_cloud_path);
            }

            // 这里直接将目录状态删除而不递归，反而更加高效且正确

            // 删除记录本地历史树和云端历史树
            TRACE_LOG("删除记录本地历史树%s和云端历史树%s", next_local_path.c_str(), next_cloud_path.c_str());
            local_history_tree->remove(next_local_path);
            cloud_history_tree->remove(next_cloud_path);
        }
        TRACE_LOG("完成处理当前项%s", filepath.c_str());
    }
    TRACE_LOG("完成本地历史元信息树的每一项");
}

// pull算法
void Synchronize::algorithm_pull(std::shared_ptr<StateBase> cloud, std::shared_ptr<StateBase> cloud_history, std::string cloud_path, std::string local_path, std::shared_ptr<StateBase> local_history)
{
    // 将StateBase指针转换为DirectoryState指针
    std::shared_ptr<DirectoryState> cloud_current_tree = std::dynamic_pointer_cast<DirectoryState>(cloud);
    std::shared_ptr<DirectoryState> cloud_history_tree = std::dynamic_pointer_cast<DirectoryState>(cloud_history);
    std::shared_ptr<DirectoryState> local_history_tree = std::dynamic_pointer_cast<DirectoryState>(local_history);

    TRACE_LOG("准备遍历云端当前元信息树的每一项，判断是否需要进行同步操作");
    // 处理云端新增、修改部分，下载更新到本地
    // 遍历处理当前目录下的子目录和子文件child
    for (auto &child : cloud_current_tree->getChildren())
    {
        // 取出child的路径
        std::string filepath = child->getFilename();
        // 对应当前child在本地的路径
        std::string next_local_path = local_path + GetFileName(filepath);
        // 对应child在云端的路径
        std::string next_cloud_path = filepath;
        // 在同一目录的历史树中查找child是否也存在
        // 找不到为NULL
        std::shared_ptr<StateBase> next_cloud_history = cloud_history_tree->findState(next_cloud_path, false, cfs, local_path, cloud_path);
        TRACE_LOG("准备处理当前项%s", filepath.c_str());
        TRACE_LOG("对应本地文件路径为%s", next_local_path.c_str());
        TRACE_LOG("对应云端文件路径为%s", next_cloud_path.c_str());
        if (next_cloud_history)
        {
            TRACE_LOG("云端历史树中存在");
        }
        else
        {
            TRACE_LOG("云端历史树中不存在");
        }

        // 如果当前child是目录
        if (child->getFiletype() == Directory)
        {
            TRACE_LOG("当前child为目录");
            // 如果child在云端历史树中存在，则递归pull
            if (next_cloud_history)
            {
                std::shared_ptr<StateBase> next_local_history = local_history_tree->findState(next_local_path, true, cfs, local_path, cloud_path);
                
                algorithm_pull(child, next_cloud_history, next_cloud_path, next_local_path, next_local_history);
            }
            // 如果child在云端历史树中不存在，则直接将该目录下载到本地
            else
            {
                subject->Set_data(CREATE_LOCAL_FOLDER, next_cloud_path, next_local_path);

                // TODO
                // 这里应该递归更新历史树，历史树的更新操作跟在最底层的操作方法中，这样效率更高
                // 这样的话代码需要重构，并且为了效率，需要重新设计代码结构
                // 目前只插入目录，没有插入目录下文件，会导致pull下载后再进行push上传，虽然用if逻辑阻断了不影响效果，但是多了函数栈的消耗

                // 插入记录到本地历史树和云端历史树
                TRACE_LOG("插入记录云端历史树%s和本地历史树%s", next_cloud_path.c_str(), next_local_path.c_str());
                cloud_history_tree->insert(std::make_shared<DirectoryState>(next_cloud_path));
                local_history_tree->insert(std::make_shared<DirectoryState>(next_local_path));
            }
        }
        // 否则child就是文件
        else
        {
            TRACE_LOG("当前child为文件");
            // 如果child在云端历史树中存在，且此本地文件与当前云端文件摘要值不相同，且云端最新，则更新本地文件
            if (next_cloud_history && std::stol(child->getMtime()) > std::stol(next_cloud_history->getMtime()) && child->getFileid() != next_cloud_history->getFileid())
            {
                subject->Set_data(UPDATE_LOCAL_FILE, next_cloud_path, next_local_path);

                // TODO
                // 把历史树的更新跟最底层的操作写在一起，可以减少不必要操作

                // 更新记录到本地历史树和云端历史树
                TRACE_LOG("更新记录云端历史树%s和本地历史树%s", next_cloud_path.c_str(), next_local_path.c_str());
                std::shared_ptr<StateBase> temp_cloud = cloud_history_tree->findState(next_cloud_path, false, cfs, local_path, cloud_path);
                if (temp_cloud)
                {
                    temp_cloud->setMtime(child->getMtime());
                    temp_cloud->setFileid(child->getFileid());
                }
                std::shared_ptr<StateBase> temp_local = local_history_tree->findState(next_local_path, true, cfs, local_path, cloud_path);
                if (temp_local)
                {
                    struct stat buf;
                    stat(next_local_path.c_str(), &buf);
                    std::string mtime = std::to_string(buf.st_mtimespec.tv_sec);
                    temp_local->setMtime(mtime);
                    temp_local->setFileid(child->getFileid());
                }
            }
            // 如果child在云端历史树中不存在，则直接将该文件下载
            if (!next_cloud_history)
            {
                subject->Set_data(DOWNLOAD_FILE, next_cloud_path, next_local_path);
                
                // TODO
                // 把历史树的更新跟最底层的操作写在一起，可以减少不必要操作

                // 插入记录到本地历史树和云端历史树
                TRACE_LOG("插入记录云端历史树%s和本地历史树%s", next_cloud_path.c_str(), next_local_path.c_str());
                cloud_history_tree->insert(std::make_shared<FileState>(next_cloud_path, child->getMtime(), child->getFileid()));
                struct stat buf;
                stat(next_local_path.c_str(), &buf);
                std::string mtime = std::to_string(buf.st_mtimespec.tv_sec);
                local_history_tree->insert(std::make_shared<FileState>(next_local_path, mtime, child->getFileid()));
            }
        }
        TRACE_LOG("完成处理当前项%s", filepath.c_str());
    }
    TRACE_LOG("完成遍历云端当前元信息树的每一项");

    // 处理云端删除部分，在本地删除
    // 如果历史树不存在，结束运行
    if (!cloud_history)
    {
        return;
    }
    TRACE_LOG("准备遍历云端历史元信息树的每一项，判断是否需要进行删除本地文件操作");
    // 遍历云端历史元信息树的每一项，判断是否需要进行本地文件删除操作
    for (auto &child : cloud_history_tree->getChildren())
    {
        // 取出child的路径
        std::string filepath = child->getFilename();
        // 对应当前child在本地的路径
        std::string next_local_path = local_path + GetFileName(filepath);
        // 对应child在云端的路径
        std::string next_cloud_path = filepath;
        // 在同一目录的历史树中查找child是否也存在
        // 找不到为NULL
        std::shared_ptr<StateBase> next_cloud_current = cloud_current_tree->findState(next_cloud_path, false, cfs, local_path, cloud_path);
        if (next_cloud_current)
        {
            TRACE_LOG("云端当前树中存在");
        }
        else
        {
            TRACE_LOG("云端当前树中不存在，在本地执行删除操作");
            if (child->getFiletype() == Directory)
            {
                subject->Set_data(DELETE_LOCAL_FOLDER, next_local_path);
            }
            else
            {
                subject->Set_data(DELETE_LOCAL_FILE, next_local_path);
            }

            // 这里直接将目录状态删除而不递归，反而更加高效且正确

            // 删除记录本地历史树和云端历史树
            TRACE_LOG("删除记录云端历史树%s和本地历史树%s", next_cloud_path.c_str(), next_local_path.c_str());
            cloud_history_tree->remove(next_cloud_path);
            local_history_tree->remove(next_local_path);
        }
        TRACE_LOG("完成处理当前项%s", filepath.c_str());
    }
    TRACE_LOG("完成遍历云端历史元信息树的每一项");
}

// 保存本地和云端历史树到磁盘
void Synchronize::save_history()
{
    TRACE_LOG("准备将本地历史元信息树写入本地磁盘");
    // 保存云端历史树
    save_statetree(config_history_path + ".cloud", metatree_cloud_history);
    // 保存本地历史树
    save_statetree(config_history_path + ".local", metatree_local_history);
    TRACE_LOG("完成将本地历史元信息树写入本地磁盘");
}

// 进行一轮同步函数
void Synchronize::synchronize()
{
    // 构建本地当前树
    INFO_LOG("准备构建本地当前状态树");
    metatree_local = generate_statetree_local(config_local_path);
    INFO_LOG("完成构建本地当前状态树");

    // 执行PUSH操作
    INFO_LOG("准备执行PUSH操作");
    algorithm_push(metatree_local, metatree_local_history, config_local_path, config_cloud_path, metatree_cloud_history);
    INFO_LOG("完成执行PUSH操作");
    
    // 构建云端当前树
    INFO_LOG("准备构建云端当前状态树");
    metatree_cloud = generate_statetree_cloud(config_cloud_path, cfs);
    INFO_LOG("完成构建云端当前状态树");

    // 执行PULL操作
    INFO_LOG("准备执行PULL操作");
    algorithm_pull(metatree_cloud, metatree_cloud_history, config_cloud_path, config_local_path, metatree_local_history);
    INFO_LOG("完成执行PULL操作");

    // 保存历史树
    INFO_LOG("完成实时更新历史状态树");
    save_history();
    INFO_LOG("保存历史树到磁盘");
}

// 云同步系统启动函数
void Synchronize::start()
{
    INFO_LOG("启动云同步系统");
    std::cout << "输入 Ctrl + C ，即可离开同步系统" << std::endl;
    // 执行start函数后close为False
    close_tag = false;
    // 初始化云同步系统
    initialize();
    while (!close_tag)
    {
        // 进行一次同步
        synchronize();
        // 设置挂起当前线程的时间（同步间隔时间）
        sleep(5);
    }
    std::cout << "关闭同步" << std::endl;
    INFO_LOG("关闭云同步系统");
}
