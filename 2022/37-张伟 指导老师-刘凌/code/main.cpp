#include <iostream>
#include "mi_command_line.h"
 
int main(int argc, char* argv[])
{
	try
	{
        MICommandLine cmd;
		return cmd.Main(argc, argv);
	}
	catch (std::exception e) 
	{
		std::cerr << e.what() << std::endl;
	}
	return -1;
}



