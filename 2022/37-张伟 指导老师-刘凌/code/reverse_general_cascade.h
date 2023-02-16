#ifndef ReverseGeneralCascade_h__
#define ReverseGeneralCascade_h__

#include <random>
#include <vector>
#include <set>
#include <cassert>
#include "common.h"
#include "graph.h"
#include "mi_random.h"

/// Container for RR set.
typedef std::vector<int> RRVec;
/// RR set with distance
typedef std::vector< std::pair<int, int> > RRDVec;

/// Template class for Reverse General Cascade
template<class TGraph=Graph>
class ReverseGCascadeT
{
public:
	typedef TGraph graph_type;

protected:
	int	n, m;
	MIRandom random;
	graph_type* gf;

	

public:
	ReverseGCascadeT() : n(0), m(0), gf(NULL) {}

public:
	void Build(TGraph& gf)
	{
		this->gf = &gf;
		this->n = gf.GetN();
		this->m = gf.GetM();
	}

	int GenRandomNode()
	{
		int id = random.RandInt(0, n-1); 
		return id;
	}

	double ReversePropagate(int num_iter, int target,
						std::vector< RRVec >& outRRSets,
                            int& outEdgeVisited)
	{
		if (gf == NULL) {
			throw NullPointerException("Please Build Graph first. (gf==NULL)");
		}

		int targetSize = 1;
		int resultSize = 0;
		outEdgeVisited = 0;
		ProbTransfom trans(gf->edgeForm);

		for (int it=0; it<num_iter; it++)
		{
            std::vector<bool> active(n, false);
			std::vector<int> RR;
			RR.push_back(target);
			active[target] = true;
			resultSize ++;

			int	h = 0;
			int t = targetSize;

			while (h<t) 
			{
				int k = gf->GetNeighborCount(RR[h]);
				for (int i=0; i<k; i++)
				{
					auto& e = gf->GetEdge(RR[h], i);

					if (active[e.v]) continue;
					outEdgeVisited++;
					if (random.RandBernoulli(trans.Prob(e.w2)))
					{
						RR.push_back(e.v);
						active[e.v] = true;
						t++;
						resultSize++;
					}
				}
				h++;
			}
			outRRSets.push_back(RR);
		}
		return (double)resultSize / (double)num_iter;
	}

	double ReversePropagate(int num_iter, int target,
                        std::vector< RRDVec >& outRRSets,
                            int& outEdgeVisited)
	{
	    if (gf == NULL) {
	        throw NullPointerException("Please Build Graph first. (gf==NULL)");
	    }

	    int targetSize = 1;
	    int resultSize = 0;
	    outEdgeVisited = 0;
	    ProbTransfom trans(gf->edgeForm);

	    for (int it=0; it<num_iter; it++)
	    {
	        std::vector<bool> active(n, false);
	        std::vector< std::pair<int, int> > RR;
	        RR.push_back(std::pair<int, int>(target, 0));
	        active[target] = true;
	        resultSize ++;

	        int h = 0;
	        int t = targetSize;

	        while (h<t) 
	        {
	            int k = gf->GetNeighborCount(RR[h].first);
	            for (int i=0; i<k; i++)
	            {
	                auto& e = gf->GetEdge(RR[h].first, i);

	                if (active[e.v]) continue;
	                outEdgeVisited++;
	                if (random.RandBernoulli(trans.Prob(e.w2)))
	                {
	                    RR.push_back(std::pair<int, int>(e.v, RR[h].second + 1));
	                    active[e.v] = true;
	                    t++;
	                    resultSize++;
	                }
	            }
	            h++;
	        }
	        outRRSets.push_back(RR);
	    }
	    return (double)resultSize / (double)num_iter;
	}
};

typedef ReverseGCascadeT<Graph> ReverseGCascade;

#endif // ReverseGeneralCascade_h__
