#include <stdio.h>
#include <string.h>
void fun1()
{
    return;
}
void fun2()
{
    return;
}
int main()
{
    char str[20];
    gets(str);
    while (1)
    {
        if (str[0] != '\0')
        {
            fun1();
            break;
        }
        else
        {
            fun2();
        }
    }
}
