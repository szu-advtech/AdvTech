#pragma once

#include "myutils.h"

#include <stdio.h>
#include <memory>

struct Colour
{
    enum Code
    {
        Default = 0,

        White,
        Red,
        Green,
        Blue,
        Cyan,
        Yellow,
        Grey,

        Bright = 0x10,

        BrightRed = Bright | Red,
        BrightGreen = Bright | Green,
        LightGrey = Bright | Grey,
        BrightWhite = Bright | White,

        // By intention
        FileName = LightGrey,
        Warning = Yellow,
        ResultError = BrightRed,
        ResultSuccess = BrightGreen,
        ResultExpectedFailure = Warning,

        Error = BrightRed,
        Success = Green,

        OriginalExpression = Cyan,
        ReconstructedExpression = Yellow,

        SecondaryText = LightGrey,
        Headers = White
    };
};

// ConsoleColourSetter类用于设置控制台输出颜色
// 在unix.cpp和win.cpp分别实现
class ConsoleColourSetter
{
public:
    DISABLE_COPY_MOVE(ConsoleColourSetter)

    explicit ConsoleColourSetter() {}
    virtual ~ConsoleColourSetter() {}
    virtual void use(Colour::Code colour) noexcept = 0;

    // Returns null if fp is not connected to console/tty
    static std::unique_ptr<ConsoleColourSetter> create_setter(FILE* fp);
};

// posix标准的控制台输出颜色设置器，win和linux都可用，win也遵守部分posix
class POSIXColourSetter final : public ConsoleColourSetter
{
public:
    explicit POSIXColourSetter(FILE* fp) : m_fp(fp) {}

    void use(Colour::Code _colourCode) noexcept override;

private:
    FILE* m_fp;
    void setColour(const char* _escapeCode) noexcept;
};