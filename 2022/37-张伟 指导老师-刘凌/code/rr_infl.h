/// rr_infl.h
/// Borgs, Christian, et al. "Maximizing social influence in nearly optimal time." 
/// Proceedings of the Twenty-Fifth Annual ACM-SIAM Symposium on Discrete Algorithms. SIAM, 2014.

#ifndef __rr_infl_h__
#define __rr_infl_h__

#include <set>
#include <vector>
#include "graph.h"
#include "common.h"
#include "reverse_general_cascade.h"
#include "algo_base.h"
#include "general_cascade.h"


/// Base class for Reverse Influence Maximization 
class RRInflBase
	: public AlgoBase
{
public:
	typedef Graph graph_type;
	typedef ReverseGCascade cascade_type;

	RRInflBase() : m(0),
			isConcurrent(false) {
	}

	// for concurrent optimization: using omp
	bool isConcurrent; // turn on to use openmp

protected:
	int m;
	
	std::vector< RRVec > table;
	std::vector<int> targets;
	// degree of hyper-edges v, where e(u, v) in the hyper graph
	// source id --> degrees
	std::vector<int> degrees;
	std::vector< std::vector<int> > degreeRRIndices;
	std::set<int> sourceSet; // all the source node ids

	void InitializeConcurrent();
	
	void _AddRRSimulation(size_t num_iter, 
		cascade_type& cascade, 
		std::vector< RRVec >& refTable, 
		std::vector<int>& refTargets);
	void _AddRRSimulation(size_t num_iter,
		cascade_type& cascade, 
		std::vector< RRVec >& refTable,
		std::vector<int>& refTargets,
		std::vector<int>& refEdgeVisited);
	
	double _RunGreedy(int seed_size, 
		std::vector<int>& outSeeds, 
		std::vector<double>& outMarginalCounts);

	void _RebuildRRIndices();
	double _EstimateInfl(const std::vector<int>& seeds, std::vector<double>& out_cumuInfl);
	void _SetResults(const std::vector<int>& seeds, const std::vector<double>& cumu_spread);

};


/// Reverse Influence Maximization
/// * Use concurrent optimization for multi-cores, turn on switch /openmp
class RRInfl : 
	public RRInflBase
{
protected:
	std::string file;
	std::string time_file;
	
public:
	RRInfl() : RRInflBase(), 
		file("rr_infl.txt"),
		time_file("time_rr_infl.txt")	
	{}
	

public:
	virtual void Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter = 1000000); // [1]
	virtual void BuildInError(graph_type& gf, int k, cascade_type& cascade, double epsilon = 0.1); // [1] 0 < epsilon < 1
	
protected:
	void _Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter = 0); // internal
	// methods for finding the solution
	double DefaultRounds(int n, int m, double epsilon = 0.2); // [1]

};

#endif ///:~ __rr_infl_h__

