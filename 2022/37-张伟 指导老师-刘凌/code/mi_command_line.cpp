#include "mi_command_line.h"

using namespace std;

std::string MICommandLine::Help()
{
	std::string help =
		"[Help]\n"
		"-h: print the help \n"
		"-g : greedy, SPM and SP1M for general ic\n"
		"-rr1 <sample_num=1000000> <k = 50> (SODA'14) reverse influence maximization algorithms.\n"

		"example: max_influence -rr1 < dm-wc.txt \n"
	;
	
	return help;
}

int MICommandLine::Main(int argc, char * argv[])
{
	// create an arguments vector to include all strings
	std::vector<std::string> argVec;
	for (int i = 0; i < argc; i++) {
		std::string param = argv[i];
		argVec.push_back(param);
	}

	return Main(argc, argVec);
}

int MICommandLine::Main(int argc, std::vector<std::string>& argv)
{
	srand((unsigned)time(NULL));
	
	if (argc <= 1) {
		std::cout << Help() << std::endl;
		return 0;
	}

	std::string arg1 = argv[1]; // the switch string like "-abc"
	std::string s;

	// for -h, print help
	s = "-h";
	if (s.compare(arg1) == 0) {
		std::cout << Help() << std::endl;
		return 0;
	}

	// the following contains switches for algorithms
	system("mkdir tmp");
	system("cd tmp");

	// create empty _running_.log to indicate running
	system("del /Q _finished_.log");
	system("echo. 2> _running_.log");

	s = "-g";
	if (s.compare(arg1) == 0) {
		GreedyAlg(argc, argv);
	}

	s = "-rr";
	if (s.compare(arg1.substr(0, 3)) == 0) {
		RRAlg(argc, argv);
	}

	// delete _running_.log to indicate finish
	system("del /Q _running_.log");
	system("echo. 2> _finished_.log");

	return 0;
}


void MICommandLine::GreedyAlg(int argc, std::vector<std::string>& argv)
{
	// GreedyGC (improved by CELF)
	GraphFactory fact;
	Graph gf = fact.Build(std::cin);
	GeneralCascade cascade;
	cascade.Build(gf);

	EventTimer timer;
	timer.SetTimeEvent("start");
	Greedy alg;
	alg.Build(gf, min(SET_SIZE, gf.GetN()), cascade);
	timer.SetTimeEvent("end");

	FILE* timetmpfile;
	fopen_s(&timetmpfile, "time_greedy.txt", "w");
	fprintf(timetmpfile, "%g\n", timer.TimeSpan("start", "end"));
	fclose(timetmpfile);

	system("del /Q tmp\\*");
}


void MICommandLine::RRAlg(int argc, std::vector<std::string>& argv)
{
	string arg1(argv[1]);

	bool isSODA14 = false;
	int num_iter = 1000000;
	int topk = SET_SIZE;
	bool isConcurrent = false;

	if (arg1.compare("-rr") == 0) {
		isSODA14 = true; 
	}
	else if (arg1.compare("-rro") == 0) {
		isSODA14 =  true; 
		isConcurrent = true;
	}
	else if (arg1.substr(0, 4).compare("-rr1") == 0) {
		isSODA14 = true;
		if (argc >= 3) num_iter = std::stoi(argv[2]);
		if (argc >= 4) topk = std::stoi(argv[3]);
		if (arg1.compare("-rr1o") == 0)
			isConcurrent = true;
	}

	GraphFactory fact;
	Graph gf = fact.Build(std::cin);
	ReverseGCascade cascade;
	cascade.Build(gf);
	

	if (isSODA14) {
		int maxK = min(topk, gf.GetN());
		cout << "=== Algorithm 1: SODA'14 ===" << endl;
		cout << "#seeds = " << maxK << endl;
		RRInfl infl;
		infl.isConcurrent = isConcurrent;
		infl.Build(gf, maxK, cascade, num_iter);
	}
}
