#include "subject.h"

void Subject::Attach(std::shared_ptr<Observer> observer) noexcept
{
    list_observer.push_back(observer);
}

void Subject::Detach(std::shared_ptr<Observer> observer) noexcept
{
    list_observer.remove(observer);
}

void Subject::Notify()
{
    std::list<std::shared_ptr<Observer>>::iterator iterator = list_observer.begin();
    while (iterator != list_observer.end())
    {
        // update不能用智能指针传参
        // 测试了用智能指针会出错，因为用this裸指针创建的shared_ptr计数器是独立的，用share_ptr复制创建shared_ptr才是累计的
        // https://heleifz.github.io/14696398760857.html
        (*iterator)->Update(this);
        ++iterator;
    }
}
