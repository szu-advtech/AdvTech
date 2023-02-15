#ifndef GREEDY_H
#define GREEDY_H
 
#include <vector>
#include <set>
#include <string>
#include "algo_base.h"
#include "common.h"
#include "graph.h"
#include "cascade.h"

/// Greedy algorithm with lazy-forward optimization
class Greedy
	: public AlgoBase
{
protected:
	std::string file;

public:
	Greedy();

	void Build(IGraph& gf, int k, ICascade& cascade);
	void BuildRanking(IGraph& gf, int k, ICascade& cascade);
	void BuildFromFile(IGraph& gf, const char* filename);
};

#endif

