#include "graph_basic.h"
#include "graph.h"

void Node::Serialize(std::ostream& sout)
{
	sout << label;
}

void Node::Deserialize(std::istream& sin)
{
	sin >> label;
}

bool Node::operator == (const Node& n) const
{
	return label == n.label;
}

// Edge
void Edge::Serialize(std::ostream& sout, IGraph& gf) {
	sout << u << "\t" << v << "\t"
		<< w1 << "\t" << w2;
}
void Edge::Deserialize(std::istream& sin, IGraph& gf) {
	Node su;
	Node sv;
	su.Deserialize(sin);
	sv.Deserialize(sin);
	u = gf.InsertNode(su);
	v = gf.InsertNode(sv);

	sin >> w1 >> w2;
}

//////////////////////////////////////////
// ContEdge
void ContEdge::Serialize(std::ostream& sout, IGraph& gf)  {
	sout << u << "\t" << v << "\t"
		<< u_a << "\t" << u_b << "\t"
		<< v_a << "\t" << v_b << "\t"
		<< w1 << "\t" << w2;
}

void ContEdge::Deserialize(std::istream& sin, IGraph& gf) {
	Node su;
	Node sv;
	su.Deserialize(sin);
	sv.Deserialize(sin);
	u = gf.InsertNode(su);
	v = gf.InsertNode(sv);

	sin >> u_a >> u_b
		>> v_a >> v_b
		>> w1 >> w2;
}
