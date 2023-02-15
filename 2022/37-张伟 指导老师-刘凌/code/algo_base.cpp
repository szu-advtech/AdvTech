#include "algo_base.h"
#include "graph.h"
 
using namespace std;

AlgoBase::AlgoBase() : n(0), top(0) {}
AlgoBase::~AlgoBase() {}

int AlgoBase::GetSeed(int i)
{
	if (i<0)
		return -1;
	if (i >= top)
		return -1;
	return list[i];
}

std::vector<int>& AlgoBase::GetSeedList()
{
	return list;
}

void AlgoBase::WriteToFile(const std::string& filename, IGraph& gf)
{
	SeedIO io;
	io.Write(filename, list, d, gf);
}

void AlgoBase::ReadFromFile(const std::string& filename, IGraph& gf)
{
	SeedIO io;
	std::vector<double> infl;
	std::vector<int> seeds = io.Read(filename, infl, gf);
	
	// make copy
	this->n = gf.GetN();
	this->list = seeds;
	this->d = infl;
	this->top = seeds.size();
};
