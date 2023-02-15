#include "synchronize.h"

int main(int argc,char *argv[])
{
    // 创建对象
    std::shared_ptr<CloudFileSystem> cfs = std::make_shared<CloudFileSystem>(argv[1]);
    std::shared_ptr<Synchronize> sync = std::make_shared<Synchronize>(cfs);
    // 开启同步
    sync->start();

    return 0;
}