#include "../sources/c_printf.h"

int main(int argc, char *argv[]) {
	c_printf(" * [r]%s\n", "Red");
	c_printf(" * [g]%s\n", "Green");
	c_printf(" * [y]%s\n", "Yellow");
	c_printf(" * [b]%s\n", "Blue");
	c_printf(" * [m]%s\n", "Magenta");
	c_printf(" * [c]%s\n", "Cyan");
	c_printf(" * [n]%s\n", "Normal");
	return 0;
}
