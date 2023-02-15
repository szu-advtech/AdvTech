#include "state.h"
#include "logger.h"
#include "myutils.h"


#include <filesystem>
#include <time.h>
#include <sys/stat.h>

// 保存状态树
void save_statetree(std::string path, std::shared_ptr<StateBase> root)
{
    std::ofstream os(path);
    cereal::XMLOutputArchive archive(os);

    archive(root);
  
}

// 读取出状态树，返回StateBase多态智能指针shared_ptr
std::shared_ptr<StateBase> load_statetree(std::string path)
{
    std::ifstream is(path);
    cereal::XMLInputArchive ar(is);
    std::shared_ptr<StateBase> temp;
    ar(temp);

    return temp;
}

// 生成本地当前状态树，返回StateBase对象
// local_path：本地数据目录
std::shared_ptr<StateBase> generate_statetree_local(std::string local_path){
    TRACE_LOG("开始生成本地当前目录状态树%s", local_path.c_str());

    // 目录的hash和mtime不起作用，所以没有必要去获取
    // 创建当前目录DirectoryState对象
    std::shared_ptr<DirectoryState> root = std::make_shared<DirectoryState>(local_path);
    
    // 遍历当前目录下的子目录和子文件，添加到当前目录的children中
    TRACE_LOG("开始遍历当前目录%s", local_path.c_str());
    for (auto&p : std::filesystem::directory_iterator(local_path))
	{   
        // 如果是目录，递归进去处理
		if (p.is_directory())
        {
            // 获得子目录路径
            auto directory = p.path().string() + "/";
            TRACE_LOG("发现子目录%s", directory.c_str());
            // 创建子目录DirectoryState对象，递归进去处理子目录
            std::shared_ptr<StateBase> subdir = generate_statetree_local(directory);
            // 将子目录DirectoryState对象插入到当前目录DirectoryState对象的children中
            root->insert(subdir);
            TRACE_LOG("将子目录%s状态插入到当前目录%s", directory.c_str(), local_path.c_str());
        }
        // 如果是文件，直接添加到当前目录children中
        else
        {
            // 获得子文件路径
            auto file = p.path().string();
            // 生成文件的mtime
            struct stat buf;
            stat(file.c_str(), &buf);
            std::string mtime = std::to_string(buf.st_mtimespec.tv_sec);
            // 计算文件的hash
            std::string fileid = CalSHA256_ByFile(file);
            TRACE_LOG("发现子文件%s，其hash为%s，mtime为%s", file.c_str(), fileid.c_str(), mtime.c_str());
            // 创建子文件FileState对象
            std::shared_ptr<StateBase> child_file = std::make_shared<FileState>(file, mtime, fileid);
            // 将子文件FileState对象插入到当前目录DirectoryState对象的children中
            root->insert(child_file);
            TRACE_LOG("将子文件%s状态插入到当前目录%s", file.c_str(), local_path.c_str());
        }
	}

    TRACE_LOG("完成生成本地当前目录状态树%s", local_path.c_str());
    return root;
}

// 生成云端当前状态树，返回StateBase对象
// cloud_path：云端数据目录
// cfs：云端文件系统对象，智能指针
std::shared_ptr<StateBase> generate_statetree_cloud(std::string cloud_path, std::shared_ptr<CloudFileSystem> cfs){
    TRACE_LOG("开始生成云端当前目录状态树%s", cloud_path.c_str());

    // 目录的hash和mtime不起作用，所以没有必要去获取
    // 创建当前目录DirectoryState对象
    std::shared_ptr<DirectoryState> root = std::make_shared<DirectoryState>(cloud_path);
    
    // 遍历当前目录下的子目录和子文件，添加到当前目录的children中
    TRACE_LOG("开始遍历当前目录%s", cloud_path.c_str());
    for (auto&p : cfs->list_files(cloud_path))
	{   
        // 如果是目录，递归进去处理
		if (p[p.size()-1] == '/')
        {
            // 获得子目录路径
            auto directory = cloud_path + p;
            TRACE_LOG("发现子目录%s", directory.c_str());
            // 创建子目录DirectoryState对象，递归进去处理子目录
            std::shared_ptr<StateBase> subdir = generate_statetree_cloud(directory, cfs);
            // 将子目录DirectoryState对象插入到当前目录DirectoryState对象的children中
            root->insert(subdir);
            TRACE_LOG("将子目录%s状态插入到当前目录%s", directory.c_str(), cloud_path.c_str());
        }
        // 如果是文件，直接添加到当前目录children中
        else
        {
            // 获得子文件路径
            auto file = cloud_path + p;
            // 获取文件的mtime和hash
            std::map<std::string, std::string> meta = cfs->stat_file(file);
            TRACE_LOG("发现子文件%s，其hash为%s，mtime为%s", file.c_str(), meta["hash"].c_str(), meta["mtime"].c_str());
            // 创建子文件FileState对象
            std::shared_ptr<StateBase> child_file = std::make_shared<FileState>(file, meta["mtime"], meta["hash"]);
            // 将子文件FileState对象插入到当前目录DirectoryState对象的children中
            root->insert(child_file);
            TRACE_LOG("将子文件%s状态插入到当前目录%s", file.c_str(), cloud_path.c_str());
        }
	}

    TRACE_LOG("完成生成云端当前目录状态树%s", cloud_path.c_str());
    return root;
}



