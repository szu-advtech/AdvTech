
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <functional>
#include <algorithm>
#include <cassert>

#include "rr_infl.h"
#include "reverse_general_cascade.h"
#include "graph.h"
#include "event_timer.h"
#include "common.h"

using namespace std;

// define a comparator for counts
struct CountComparator
{
public:
	vector<int>& counts;
	CountComparator(vector<int>& c) :counts(c){}
public:
	bool operator () (int a, int b) {
		return (counts[a] < counts[b]);
	}
};

struct dCountComparator
{
public:
	vector<double>& counts;
	dCountComparator(vector<double>& c) :counts(c){}
public:
	bool operator () (double a, double b) {
		return (counts[a] < counts[b]);
	}
};


void RRInflBase::InitializeConcurrent()
{
	if (isConcurrent) 
	{
#ifdef MI_USE_OMP
		// run concurrently
		const double DYNAMIC_RATIO = 0.25;
		int maxThreads = omp_get_max_threads();
		omp_set_num_threads(maxThreads);
		int dynamicThreads = (int)(maxThreads * DYNAMIC_RATIO);
		omp_set_dynamic(dynamicThreads);
		
		cout << "== Turn on omp optimization: == " << endl;
		cout << "#Max Threads = " << omp_get_max_threads() << "\t#Dynamic Threads = " << omp_get_dynamic() << endl;
#else
        cout << "== omp is not supported or enabled == " << endl;
#endif
    }
}


void RRInflBase::_AddRRSimulation(size_t num_iter, 
							cascade_type& cascade, 
							std::vector< RRVec >& refTable,
							std::vector<int>& refTargets)
{
	vector<int> edgeVisited; // discard
	_AddRRSimulation(num_iter, cascade, refTable, refTargets, edgeVisited);
}

void RRInflBase::_AddRRSimulation(size_t num_iter, 
							  cascade_type& cascade, 
							  std::vector< RRVec >& refTable,
							  std::vector<int>& refTargets,
							  std::vector<int>& refEdgeVisited)
{
#ifdef MI_USE_OMP
	if (!isConcurrent) {
#endif
		// run single thread

		for (size_t iter = 0; iter < num_iter; ++iter) {
			int id = cascade.GenRandomNode();
			int edgeVisited;
			cascade.ReversePropagate(1, id, refTable, edgeVisited);

			refTargets.push_back(id);
			refEdgeVisited.push_back(edgeVisited);
		}

#ifdef MI_USE_OMP
	} else {
		// run concurrently
		#pragma omp parallel for ordered
		for (int iter = 0; iter < num_iter; ++iter) {
			int id = cascade.GenRandomNode();
			int edgeVisited;
			vector<RRVec> tmpTable;
			cascade.ReversePropagate(1, id, tmpTable, edgeVisited);
			#pragma omp critical
			{
				refTable.push_back(tmpTable[0]);
				refTargets.push_back(id);
				refEdgeVisited.push_back(edgeVisited);
			}
		}
	}
#endif

}

void RRInflBase::_RebuildRRIndices()
{
	degrees.clear();
	degrees.resize(n, 0);
	degreeRRIndices.clear();

	for (int i = 0; i < n; ++i) {
		degreeRRIndices.push_back( vector<int>() );
	}
	
	// to count hyper edges:
	for (size_t i = 0; i < table.size(); ++i) {
		const RRVec& RR = table[i];
		for (int source : RR) {
			degrees[source]++;
			degreeRRIndices[source].push_back(i); // add index of table
		}
	}

	// add to sourceSet where node's degree > 0
	sourceSet.clear();
	for (size_t i = 0; i < degrees.size(); ++i) {
		if (degrees[i] >= 0) {
			sourceSet.insert(i);
		}
	}
}


// Apply Greedy to solve Max Cover
double RRInflBase::_RunGreedy(int seed_size, 
						vector<int>& outSeeds,
						vector<double>& outEstSpread) 
{
	outSeeds.clear();
	outEstSpread.clear();

	// set enables for table
	vector<bool> enables;
	enables.resize(table.size(), true);

	set<int> candidates(sourceSet);
	CountComparator comp(degrees);

	double spread = 0;
	for (int iter = 0; iter < seed_size; ++iter) {
		set<int>::const_iterator maxPt = max_element(candidates.begin(), candidates.end(), comp);
		int maxSource = *maxPt;
		assert(degrees[maxSource] >= 0);

		// selected one node
		outSeeds.push_back(maxSource);

		// estimate spread
		spread = spread + ((double) n * degrees[maxSource] / table.size() );

		outEstSpread.push_back(spread);

		// clear values
		candidates.erase(maxPt);
		degrees[maxSource] = -1;

		// deduct the counts from the rest nodes
		const vector<int>& idxList = degreeRRIndices[maxSource];
		if (!isConcurrent) {
			for (int idx : idxList) {
				if (enables[idx]) {
					const RRVec& RRset = table[idx];
					for (int rr : RRset) {
						if (rr == maxSource) continue;
						degrees[rr]--; // deduct
					}
					enables[idx] = false;
				}
			}
		} else {
			// run concurrently
			#pragma omp parallel for
			for (int idxIter = 0; idxIter < idxList.size(); ++idxIter) {
				int idx = idxList[idxIter];
				if (enables[idx]) {
					const RRVec& RRset = table[idx];
					for (int rr : RRset) {
						if (rr == maxSource) continue;
						
						#pragma omp atomic
						degrees[rr]--; // deduct
					}
					enables[idx] = false;
				}
			}
		}
	}
    
	assert(outSeeds.size() == seed_size);
	assert(outEstSpread.size() == seed_size);
	
	return spread;
}


vector<int> vectors_intersection(vector<int> v1, vector<int> v2){
	vector<int> v;
	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));//  inter set 
	return v;
}

void RRInflBase::_SetResults(const vector<int>& seeds, const vector<double>& cumu_spread)
{
	for (int i = 0; i < top; ++i) {
		list[i] = seeds[i];
		// d[i] = cumu_spread[i];  // use cumulative spread
		d[i] = (i > 0) ? (cumu_spread[i] - cumu_spread[i - 1]) : cumu_spread[i]; // use marginal spread
	}
}

// Same thing has been implemented as a part of _Greedy
double RRInflBase::_EstimateInfl(const vector<int>& seeds, vector<double>& out_cumu_spread)
{
	set<int> covered;
	double spd = 0;

	vector<bool> enables;
	enables.resize(table.size(), true);
	for (size_t i = 0; i < seeds.size(); ++i) {
		int sd = seeds[i];
		for (int idx : degreeRRIndices[sd]) {
			if (enables[idx]) {
				covered.insert(idx);
				enables[idx] = false;
			}
		}
		spd = (double)(n * covered.size()) / table.size();
		out_cumu_spread.push_back(spd);
	}
	return spd;
}


// RRInfl
double RRInfl::DefaultRounds(int n, int m, double epsilon)
{
	return max(144.0 * (n + m) / pow(epsilon, 3) * log( max(n,1) ), 1.0); // to make it positive
}

void RRInfl::BuildInError(graph_type& gf, int k, cascade_type& cascade, double epsilon/*=0.1*/)
{
	n = gf.GetN();
	m = gf.GetM();
	size_t num_iter = (size_t)(ceil(DefaultRounds(n, m, epsilon)));
	_Build(gf, k, cascade, num_iter);
}

void RRInfl::Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter/*=1000000*/)
{
	n = gf.GetN();
	m = gf.GetM();
	_Build(gf, k, cascade, num_iter);
}
	

void RRInfl::_Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter)
{
	InitializeConcurrent();

	n = gf.GetN();
	m = gf.GetM();

	top = k;
	d.resize(top, 0.0);
	list.resize(top, 0);
	cascade.Build(gf);

	cout << "#round = " << num_iter << endl;

	table.clear();
	targets.clear();

	EventTimer pctimer;
	pctimer.SetTimeEvent("start");

	// Step 1:
	_AddRRSimulation(num_iter, cascade, table, targets);
	assert(targets.size() == num_iter);
	assert(table.size() == num_iter);
	pctimer.SetTimeEvent("step1");

	// Step 2:
	vector<int> seeds;
	vector<double> est_spread;
	_RebuildRRIndices();
	pctimer.SetTimeEvent("step2");

	// Step 3:
	double spd = _RunGreedy(k, seeds, est_spread);
	_SetResults(seeds, est_spread);
	pctimer.SetTimeEvent("end");

	cout << "  final (estimated) spread = " << spd << "\t round = " << num_iter << endl;
	
	WriteToFile(file, gf);

	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile,"%g\n", pctimer.TimeSpan("start", "end"));
	fprintf(timetmpfile,"Gen graph: %g\n", pctimer.TimeSpan("start", "step1"));
	fprintf(timetmpfile,"Build RR: %g\n", pctimer.TimeSpan("step1", "step2"));
	fprintf(timetmpfile,"Greedy: %g\n", pctimer.TimeSpan("step2", "end"));
	fclose(timetmpfile);
}
