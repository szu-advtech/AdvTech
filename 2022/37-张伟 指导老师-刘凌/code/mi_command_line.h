#ifndef mi_command_line_h__
#define mi_command_line_h__
 
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <set>
#include <fstream>
#include <sstream>
#include <random>

#include "common.h"
#include "graph.h"
#include "greedy.h"
#include "event_timer.h"

#include "rr_infl.h"
#include "cascade.h"
#include "general_cascade.h"


/// Command line for a set of max_influence algorithms
class MICommandLine
{
public:
	int Main(int argc, char* argv[]);
	int Main(int argc, std::vector<std::string>& argv);
	std::string Help();	
	void GreedyAlg(int argc, std::vector<std::string>& argv);
	void RRAlg(int argc, std::vector<std::string>& argv);
};


#endif // mi_command_line_h__
