#include "color.h"

#include <unistd.h>


// 针对unix系统的实现方法，windows想实现的话可以利用宏定义分文件定义

std::unique_ptr<ConsoleColourSetter> ConsoleColourSetter::create_setter(FILE* fp)
{
    if (!fp || !::isatty(::fileno(fp)))
        return {};
    return std::make_unique<POSIXColourSetter>(fp);
}

void POSIXColourSetter::use(Colour::Code _colourCode) noexcept
{
    switch (_colourCode)
    {
    case Colour::Default:
        return setColour("[0;39m");
    case Colour::White:
        return setColour("[0m");
    case Colour::Red:
        return setColour("[0;31m");
    case Colour::Green:
        return setColour("[0;32m");
    case Colour::Blue:
        return setColour("[0:34m");
    case Colour::Cyan:
        return setColour("[0;36m");
    case Colour::Yellow:
        return setColour("[0;33m");
    case Colour::Grey:
        return setColour("[1;30m");

    case Colour::LightGrey:
        return setColour("[0;37m");
    case Colour::BrightRed:
        return setColour("[1;31m");
    case Colour::BrightGreen:
        return setColour("[1;32m");
    case Colour::BrightWhite:
        return setColour("[1;37m");

    default:
        break;
    }
}

void POSIXColourSetter::setColour(const char* _escapeCode) noexcept
{
    putc('\033', m_fp);
    fputs(_escapeCode, m_fp);
}

