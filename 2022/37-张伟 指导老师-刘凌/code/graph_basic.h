#ifndef __GRAPH_BASIC_H__
#define __GRAPH_BASIC_H__

 
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <memory>
#include <string>
#include <cassert>
#include <type_traits>
#include "common.h"
#include "mi_random.h"


class IGraph;


class Node
{
public:
	/// Node's label (managed memory)
	std::string label;

public:
	virtual ~Node() {}

	void Serialize(std::ostream& sout);
	void Deserialize(std::istream& sin);

	/// Check equivalent
	bool operator == (const Node& n) const;
};

/// Discrete Edge (base class)
class EdgeBase
{
public:
	/// u is the left node
	int u;
	/// v is the right node
	int v;
	EdgeBase(int u = -1, int v = -1) : u(u), v(v) {}
	virtual ~EdgeBase() {}
};

/// Discrete edge. 
class Edge :
	public EdgeBase
{
public:
	/// count of its duplication
	int c;
	/// Probability u->v
	double w1;
	/// Probability v->u
	double w2;

	Edge(double w1 = 0.0, double w2 = 0.0, int c = 1) : EdgeBase(), c(c), w1(w1), w2(w2) {}

public:
	void Serialize(std::ostream& sout, IGraph& gf);
	void Deserialize(std::istream& sin, IGraph& gf);

};


/// Continuous Edge of weibull distribution
class ContEdge :
	public Edge
{
public:
	double u_a, u_b, v_a, v_b;
	ContEdge() : Edge(), u_a(0), u_b(0), v_a(0), v_b(0) {}

public:
	void Serialize(std::ostream& sout, IGraph& gf);
	void Deserialize(std::istream& sin, IGraph& gf);
};



typedef int EdgeFormType;
enum EdgeForm
{
	NORMAL_EDGE = 0, LOG_NEG_EDGE = 1
};


/// w => w
inline double identicalConv(double w) { return w; }
/// w => exp(-w)
inline double negExpConv(double w) { return exp(-w); }
/// w => max(w, 1e-32)
inline double maxPosConv(double w) {
	static double INF_POS_MIN = 1e-32;
	return (w > INF_POS_MIN ? w : INF_POS_MIN);
}
/// w => -log(w)
inline double logNegConv(double w) { return - log(maxPosConv(w)); }

/// Function pointer type
typedef double(*probFunc1)(double);


/// Class for probability transform.
/// It uses EdgeFormType as switch to know how to get probability or 
struct ProbTransfom
{
private:
	probFunc1 f; /// internal function to get Probability p
	probFunc1 logNegf; /// internal function to get -log(p)
	probFunc1 negExpf; /// internal function to get exp(-p)

public:
	/// Construct by the EdgeFormType
	ProbTransfom(EdgeFormType edgeForm)
	{
		if (edgeForm == EdgeForm::NORMAL_EDGE) {
			f = identicalConv;
			logNegf = logNegConv;
		}
		else if (edgeForm == EdgeForm::LOG_NEG_EDGE) {
			f = negExpf;
			logNegf = identicalConv;
		}
		else {
			throw std::invalid_argument("EdgeFormType not support!");
		}
	}

public:
	/// Get Probability
	inline double Prob(double w) {
		return f(w);
	}
	/// Get the negative logrithmic of probability
	inline double LogNegProb(double w) {
		return logNegf(w);
	}
};


#endif ///:~ __GRAPH_BASIC_H__
