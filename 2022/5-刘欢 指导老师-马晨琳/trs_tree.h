#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>

using std::pair;
using std::vector;
using std::queue;
using std::set;
using std::unordered_map;


namespace TRS {

extern int k;  // node_fanout
// default: 4
extern int error_bound;  // in paper: the expected number of host column N values covered by the range returned by the range returned from searching a TRS-Tree node for a point query on the tagget column M.
// default: 0.85
extern double outliers_ratio;
// TODO: control the tree's max_height
extern int max_height;

pair<int, int> intersect(pair<int, int> a, pair<int, int> b);

class Node {
private:
  vector<Node*> navi;
  int lower, upper;
  unordered_map<int, int> outliers;  // map M to main index
  double a, b, e;  // ax + b +(-) e
  bool is_leaf_node;
  int n;  // the number of the cover points

public:
  Node();

  void split();
  Node* get_sub_node(int id);
  pair<int, int> get_host_range(pair<int, int> req);
  void compute(vector<vector<int> >& tmp_table);
  bool is_over_lapping(pair<int, int>& P);
  pair<int, int> range();
  unordered_map<int, int>& get_outliers();
  void set_leaf_node(bool _is_leaf_node);
  bool query_leaf_node();

};

class TRS_Tree {
private:
  Node* root;

public:

  TRS_Tree();
  void build(vector<vector<int> >& table);
  bool validate(vector<vector<int> >& tmp_table, Node* node);
  pair<set<int>, set<int> > lookup(pair<int, int>& pre);
  void insert(int target, int sid, int tid);
  void _delete(int target);
  Node* traverse(Node* node, int target);
  long long int calMem();
};

}