#ifndef __graph_h__
#define __graph_h__

 
#include "graph_basic.h"
#include <unordered_map>
#include <string>
#include <cassert>
#include <type_traits>
#include "limits.h"
#include "common.h"
#include "mi_random.h"


inline bool isEmptyOrCommentLine(const std::string& line)
{
	return (line.empty()) || (line[0] == '#');
}

/// Interface of basic graph (abstract base class)
class IGraph
{
public:
	typedef IGraph self_type;

	virtual int GetN() const = 0;
	virtual int GetM() const = 0;

	virtual std::string MapIndexToNodeName(int idx) const = 0;
	virtual int MapNodeNameToIndex(const std::string& name) const = 0;
	virtual int InsertNode(Node& u) = 0;
};


/// Map from index to string, or from string to index
class NodeMap
{
public:
	typedef std::unordered_map<std::string, int> MapS2I;
public:
	/// map name to index
	MapS2I S2I;
	std::vector<Node> nodes;
};




/// The introduction of Graph is to incorporate multiple 
/// classes of Edge, when we deal with different diffusion models
template <class TEdge=Edge>
class GraphT
	: public IGraph
{
public:
	typedef GraphT<TEdge> self_type;
	typedef TEdge edge_type;
	
public:
	// The following are override of members of base type
	/// n is the node count
	int n;
	/// m is the edge count (directed edges)
	int m;

	std::vector<int> degree;
	std::vector<int> index;
	std::vector<TEdge> edges;
	EdgeFormType edgeForm;

protected:
	NodeMap nodeMap;

public:
	std::string MapIndexToNodeName(int idx) const { return nodeMap.nodes.at(idx).label; }
	int MapNodeNameToIndex(const std::string& name) const { return nodeMap.S2I.at(name); }
	int InsertNode(Node& u) 
	{
		const std::string& s = u.label;
		int cur_index;
		// find if key is existed
		if (!nodeMap.S2I.count(s)) {
			// not existed, create the key and insert current node
			cur_index = nodeMap.nodes.size(); // current index points to the end() of nodes
			nodeMap.nodes.push_back(u);
			nodeMap.S2I[s] = cur_index; // add to S2I
		}
		else {
			// return the index
			cur_index = nodeMap.S2I[s];
		}
		return cur_index;
	}
	/// The real nodes from the input (possibly it is <= n for sparse graph)
	int GetRealNodeCount() {
		return nodeMap.nodes.size();
	}

public:
	GraphT() : n(0), m(0), degree(), index(), edges(), nodeMap(), edgeForm(EdgeForm::NORMAL_EDGE) {}

public:
	/// Number of nodes graph
	int GetN() const { return n; }
	/// Number of (directed) edges in graph
	int GetM() const { return m; }
	virtual int GetDegree(int node) const {
        return degree[node];
    }
	virtual int GetNeighborCount(int node)
	{
		if (node == 0)
			return index[node]+1;
		else 
			return index[node]-index[node-1];
	}
	virtual edge_type& GetEdge(int node, int idx)
	{
		if (node == 0)
			return edges[idx];
		else
			return edges[index[node-1]+1+idx];
	}
};

/// Class for graph io
template<class TGraph, class TEdge>
class GraphIO
{
public:
	typedef TGraph graph_type;
	typedef TEdge edge_type;

protected:
	void ReadNM(std::istream& sin, int& out_n, int& out_m)
	{
		sin >> out_n >> out_m;
	}

	void ReadEdge(std::istream& sin, edge_type& e, graph_type& gf)
	{
		e.Deserialize(sin, gf);
	}

public:
	graph_type Read(std::istream& in)
	{
		graph_type g;
		std::string line;
		// indicate finish reading n and m.
		bool isNMFinished = false;
		// i is the counter of current edges
		int i = 0;
		int iMax = 0;

		while (getline(in, line)) {
			// skip empty line or comment line
			if (isEmptyOrCommentLine(line)) {
				continue;
			}

			if (!isNMFinished) {
				std::stringstream ssline(line);
				ReadNM(ssline, g.n, g.m);
				assert(g.n >= 0 && g.m >= 0);
				iMax = 2*g.m;
				g.degree.resize(g.n, 0);
				g.edges.resize(iMax);
				isNMFinished = true;
				continue;
			}

			if (i < iMax)
			{
				TEdge e;
				std::stringstream ssline(line);
				ReadEdge(ssline, e, g);

				g.edges[i] = e;
				g.degree[g.edges[i].u]++;
				i++;
			}
			else
			{
				break;
			}
		}
		// If there are some nodes never occurred in the edge list (we don't know their names),
		// then we name them as "$ISOLATED_NODE$_i" where i = 1, 2, 3, ...
		for (int i = 1; g.GetRealNodeCount() < g.n; i++)
		{
			Node isolated;
			isolated.label.append(ISOLATED_NODE_PREFIX);
			isolated.label.append(std::to_string(i));
			g.InsertNode(isolated);
		}

		if (i != iMax) {
			std::string ss = "Graph input is incorrect! Expect #edges = ";
			ss += std::to_string(iMax);
			ss += ", find #edges = ";
			ss += std::to_string(i);
			throw InvalidInputFormatException(ss);
		}

		return g;
	}
};

/// Compare edge based on the tuple of index (u, v)
template<class TEdge = Edge>
class EdgeComparatorT
{
public:
	typedef TEdge edge_type;
	
public:
	int compare(const edge_type& a, const edge_type& b) {
		if ((a.u > b.u) || ((a.u == b.u) && (a.v > b.v))) {
			return 1;
		}
		else if ((a.u < b.u) || ((a.u == b.u) && (a.v < b.v))) {
			return -1;
		}
		else {
			return 0;
		}
	}
};

/// Qsort algorithm for edges.
template<class TEdge = Edge>
class EdgeSorterT
{
public:
	typedef EdgeSorterT<TEdge> self_type;
	typedef TEdge edge_type;
	typedef EdgeComparatorT<TEdge> TComparator;
		
private:
	EdgeComparatorT<TEdge> comp;

public:
	EdgeSorterT() : comp() {}

	void sort(std::vector<edge_type>& edges, int h, int t) {
		_qsort(edges, h, t);
	}

protected:
	void _qsort(std::vector<edge_type>& edges, int h, int t) {
		if (h < t)
		{
			int i = h, j = t;
			edge_type mid = edges[(i + j) / 2]; // copy
			edges[(i + j) / 2] = edges[i];

			while (i < j)
			{

				while ((i < j) && (comp.compare(edges[j], mid) == 1))
					j--;
				if (i < j) {
					edges[i] = edges[j];
					i++;
				}
				while ((i < j) && (comp.compare(edges[i], mid) == -1))
					i++;
				if (i < j) {
					edges[j] = edges[i];
					j--;
				}
			}

			edges[i] = mid;
			_qsort(edges, h, i - 1);
			_qsort(edges, i + 1, t);
		}
	}
};

/// Use factory pattern to generate graphs
template<class TEdge,
		class TGraph=GraphT<TEdge>,
		class TEdgeSorter=EdgeSorterT<TEdge> >
class GraphFactoryT
{
public:
	typedef GraphFactoryT<TEdge, TGraph, TEdgeSorter> self_type;
	typedef GraphIO<TGraph, TEdge> io_type;
	typedef TGraph graph_type;
	typedef TEdge edge_type;
	typedef TEdgeSorter sorter_type;

public:
	GraphFactoryT() {}

	// Build2WC
	virtual TGraph Build(std::istream& sin)
	{
   		io_type io;

		TGraph g = io.Read(sin);
		
		// After read (sort, and build index)
		// This part can be refined by using vector edges 
		sorter_type sorter;
		sorter.sort(g.edges, 0, 2*g.m - 1);

		int m1 = 0;
		for (int i = 1; i < 2 * g.m; i++)
		{
			if ((g.edges[i].u != g.edges[m1].u) || (g.edges[i].v != g.edges[m1].v))
			{
				m1++;
				g.edges[m1] = g.edges[i];
			}
			else
			{
				g.edges[m1].c++;
			}
		}
		// reset g.m to its real edge number
		if (g.m != 0)
			g.m = m1 + 1;

		g.index.clear();
		g.index.resize(g.n);
		for (int i = 0; i < g.n; i++)
			g.index[i] = 0;
		for (int i = 0; i < g.m; i++)
			g.index[g.edges[i].u] = i;
		for (int i = 1; i < g.n; i++)
			if (g.index[i] < g.index[i - 1])
				g.index[i] = g.index[i - 1];

		return g;
	}
};



typedef GraphT<Edge>  Graph;
typedef GraphT<ContEdge>  ContGraph;
typedef GraphFactoryT<Edge> GraphFactory;
typedef GraphFactoryT<ContEdge> ContGraphFactory;


/// For Seed io
class SeedIO
{
public:
	/// ----- Seed file format ------- <br/>
	/// n <br/>
	/// seed1 ... <br/>
	/// seed2 ... <br/>
	/// seed3 ... <br/>
	/// seedn ... <br/>
	/// <br/>
	/// ------------------------------ <br/>
	/// The file allows comments anywhere in the following. <br/>
	/// # this is a comment, which starts with "#" <br/>
	std::vector<int> Read(const std::string& filename, IGraph& gf);
	std::vector<int> Read(std::istream& in, IGraph& gf);

	/// ----- Seed file format ------- <br/>
	/// n <br/>
	/// seed1 infl1 ... <br/>
	/// seed2 infl2 ... <br/>
	/// seed3 infl3 ... <br/>
	/// seedn infl4 ... <br/>
	/// <br/>
	/// ------------------------------ <br/>
	/// The file allows comments anywhere in the following. <br/>
	/// # this is a comment, which starts with "#"
	std::vector<int> Read(const std::string& filename, std::vector<double>& out_infl, IGraph& gf);
	std::vector<int> Read(std::istream& in, std::vector<double>& out_infl, IGraph& gf);

	/// ------ output format ------ <br/>
	/// n <br/>
	/// seed1 infl1 ... <br/>
	/// seed2 infl2 ... <br/>
	/// seed3 infl3 ... <br/>
	/// seedn infl4 ... <br/>
	void Write(const std::string& filename,
		const std::vector<int>& seeds,
		const std::vector<double>& infl, IGraph& gf);

	void Write(std::ostream& out,
		const std::vector<int>& seeds,
		const std::vector<double>& infl, IGraph& gf);
};

#endif // __graph_h__
