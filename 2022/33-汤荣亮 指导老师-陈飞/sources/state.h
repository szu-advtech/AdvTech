#pragma once

#include "cfs.h"
#include "myutils.h"

#include <cereal/types/memory.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <fstream>
#include <cereal/types/list.hpp>
#include <iostream>
#include <cereal/types/polymorphic.hpp>
#include <string.h>

/*
TODO
比较list和vector在大量文件时的性能。
因为将历史树的更新改为实时更新后，插入删除的操作更加频繁（构建当前树+更新历史树），所以文件状态对象改为list容器存放。

children是否需要改为list类型，因为vector为连续存储，当文件特别多，对象占用空间很大时怕没有连续空间放vector。
list和vector性能简单比较：https://blog.csdn.net/xxm524/article/details/86561828
*/

// 枚举文件类型：文件、目录
enum FileType
{
    File,
    Directory
};

// 状态树的基类
class StateBase
{
private:
    // 把类声明序列化函数放在private或protect域下面的时候，需要用access，这样cereal才有对该类的访问权
    friend class cereal::access;
    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(file_type, filename, mtime, file_id);
    }

private:
    // 文件类型
    FileType file_type;
    // 文件名
    std::string filename;
    // 修改时间
    std::string mtime;
    // 文件的hash
    std::string file_id;

public:
    StateBase() {}
    // 基类要有虚函数，编译器才知道用到了多态，dynamic和cereal都有用
    virtual ~StateBase() {}
    explicit StateBase(FileType type, std::string name, std::string time, std::string id) : file_type(type), filename(name), mtime(time), file_id(id) {}

    void setFiletype(FileType type) noexcept
    {
        file_type = type;
    }
    FileType getFiletype() noexcept
    {
        return file_type;
    }
    void setFilename(std::string name) noexcept
    {
        filename = name;
    }
    std::string getFilename() noexcept
    {
        return filename;
    }
    void setMtime(std::string time) noexcept
    {
        mtime = time;
    }
    std::string getMtime() noexcept
    {
        return mtime;
    }
    void setFileid(std::string id) noexcept
    {
        file_id = id;
    }
    std::string getFileid() noexcept
    {
        return file_id;
    }
};

// 文件状态
class FileState : public StateBase
{
private:
    // 把类声明序列化函数放在private或protect域下面的时候，需要用access，这样cereal才有对该类的访问权
    friend class cereal::access;
    template <class Archive>
    void serialize( Archive & ar )
    {
        // 序列化基类信息
        ar(cereal::base_class<StateBase>(this)); 
    }

public:
    FileState() {}
    explicit FileState(std::string name, std::string time = "", std::string id = "") : StateBase(File, name, time, id) {}
};

// 目录状态（状态树）
class DirectoryState : public StateBase
{
private:
    // 把类声明序列化函数放在private或protect域下面的时候，需要用access，这样cereal才有对该类的访问权
    friend class cereal::access;
    template <class Archive>
    void serialize( Archive & ar )
    {
        // 序列化基类信息
        ar(cereal::base_class<StateBase>(this), children); 

        // we have to explicitly inform the archive when it is safe to serialize
        // the deferred data - this should only be called once on the archive
        ar.serializeDeferments();
    }

private:
    // 存放当前directory的孩子节点
    std::list<std::shared_ptr<StateBase>> children;
    // 定义别名
    typedef std::list<std::shared_ptr<StateBase>> list;

public:
    DirectoryState() {}
    explicit DirectoryState(std::string name, std::string time = "", std::string id = "") : StateBase(Directory, name, time, id) {}
    // explicit DirectoryState(vector ch, std::string name, std::string time = "", std::string id = "") : StateBase(Directory, name, time, id), children(ch) {}

    // void setChildren(vector ch)
    // {
    //     children = ch;
    // }
    // 获取所有孩子节点
    list getChildren() noexcept
    {
        return children;
    }
    // 在children中添加一条记录
    void insert(std::shared_ptr<StateBase> st) noexcept
    {
        children.push_back(st);
    }
    // 将文件名为filename的记录从children中删掉
    void remove(std::string name) noexcept
    {
        // 声明一个迭代器
        std::list<std::shared_ptr<StateBase>>::iterator it;
        for(it = children.begin();it!=children.end();it++){
            if((*it)->getFilename() == name){
                children.erase(it);
            }
        }
    }
    // 根据goal（filename, file_id）从children找出对应的对象，返回指针，找不到返回空指针
    // 本地和云端不区分大小写，如果本地发现大小写不一致，则更新云端数据的大小写，保证云端与本地一致
    std::shared_ptr<StateBase> findState(std::string goal, bool isLocalTree, std::shared_ptr<CloudFileSystem> cfs, std::string local_path, std::string cloud_path) noexcept
    {
        // 声明一个迭代器
        std::list<std::shared_ptr<StateBase>>::iterator it;
        for(it = children.begin();it!=children.end();it++){            
            // 忽略大小写，看filename是否相等
            // strcasecmp函数为linux下不区分大小写的比较函数，在string.h头文件中
            std::string fileName = (*it)->getFilename();
            if (strcasecmp(goal.c_str(), fileName.c_str()) == 0)
            {
                // 如果比较大小写，不相等的话需要重设找到的状态的filename
                // 目的是为了保证历史树中的状态的filename及时更新
                if(fileName != goal){
                    // 如果是本地的状态树，说明本地出现了大小写不一致的情况，需要同步云端的数据大小写与本地一致
                    if (isLocalTree)
                    {
                        std::string old_cloud_path = cloud_path + GetFileName(fileName);
                        std::string new_cloud_path = cloud_path + GetFileName(goal);
                        std::cout << "旧云端路径：" << old_cloud_path << std::endl;
                        std::cout << "新云端路径：" << new_cloud_path << std::endl;
                        // TODO
                        // (*it)->setFilename(goal);
                        // 重命名状态树中目录后，状态树目录内的文件并未重命名，会递归对这些文件进行重复操作，这里需要优化下
                        cfs->rename(old_cloud_path, new_cloud_path);
                    }
                    else
                    // 否则就是云端的状态树，说明云端出现了大小写不一致的情况（另一个设备更新大小写引起），需要同步本地的数据大小写与云端一致
                    {
                        std::string old_local_path = local_path + GetFileName(fileName);
                        std::string new_local_path = local_path + GetFileName(goal);
                        std::cout << "旧本地路径：" << old_local_path << std::endl;
                        std::cout << "新本地路径：" << new_local_path << std::endl;
                        // TODO
                        // (*it)->setFilename(goal);
                        // 重命名状态树中目录后，状态树目录内的文件并未重命名，会递归对这些文件进行重复操作，这里需要优化下
                        // host1本地完成更改大小写后，会对host1进行重复无效rename，这里也需要优化
                        std::filesystem::rename(old_local_path, new_local_path);
                    }

                    (*it)->setFilename(goal);
                }
                return *it;
            }
            // 根据hash来找状态
            if ((*it)->getFileid() == goal)
            {
                return *it;
            }
        }
        
        return NULL;
    }
};

// 登记用到的子类，父类可以不用登记
CEREAL_REGISTER_TYPE(FileState);
CEREAL_REGISTER_TYPE(DirectoryState);
// 登记继承关系
CEREAL_REGISTER_POLYMORPHIC_RELATION(StateBase, FileState)
CEREAL_REGISTER_POLYMORPHIC_RELATION(StateBase, DirectoryState)

// 保存状态树
void save_statetree(std::string path, std::shared_ptr<StateBase> root);

// 读取出状态树，返回StateBase多态智能指针shared_ptr
std::shared_ptr<StateBase> load_statetree(std::string path);

// 生成本地当前状态树
std::shared_ptr<StateBase> generate_statetree_local(std::string local_path);

// 生成云端当前状态树
std::shared_ptr<StateBase> generate_statetree_cloud(std::string cloud_path, std::shared_ptr<CloudFileSystem> cfs);
