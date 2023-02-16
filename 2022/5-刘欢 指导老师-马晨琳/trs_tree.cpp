#include "trs_tree.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#define Log std::cout
using std::endl;
using std::pair;
using std::vector;
using std::queue;
using std::string;
using std::unordered_map;
using std::max;
using std::min;

namespace TRS {

int k = 8;  // k-ary tree node (maximum of navi)
int error_bound = 2;  // in paper: the expected number of host column N values covered by the range returned by the range returned from searching a TRS-Tree node for a point query on the tagget column M.
double outliers_ratio = 0.1;
// TODO: control the tree's max_height
int max_height = 10;

pair<int, int> intersect(pair<int, int> a, pair<int, int> b) {  // range pre
  int la = a.first;
  int ua = a.second;
  int lb = b.first;
  int ub = b.second;
//  Log << "[trs_tree] (" << la << ", " << ua << ") " << "相交" << " (" << lb << ", " << ub << ")" << endl;
  if (ua < lb || ub < la) {
    return {0, 0};
  }
  int lower = max(la, lb);
  int upper = min(ua, ub);
//  Log << "[trs_tree] 相交结果: " << "(" << lower << ", " << upper << ")" << endl;
  return { lower, upper };
}

Node::Node() {
  is_leaf_node = true;
  n = 0;
  lower = 0x7fffffff;
  upper = 0x80000000;
}

void Node::split() {
  if (navi.size() == k) {
    return;
  }
  for (int i = 0; i < k; i += 1) {
    navi.push_back(new Node);
  }
}

Node* Node::get_sub_node(int id) {
  return navi[id];
}

pair<int, int> Node::get_host_range(pair<int, int> req) {
  int lb = req.first;
  int ub = req.second;
  int min, max;
  if (a >= 0) {
    min = int(a * lb + b - e);
    max = int(a * ub + b + e);
  }
  else {
    max = int(a * lb + b + e);
    min = int(a * ub + b - e);
  }
//  Log << "[node] " << "(" << req.first << ", " << req.second << ")" << " 的线性估算范围为 " << "(" << min << ", " << max << ")" << endl;
  return { min, max };
}

// M ==> N
void Node::compute(vector<vector<int> >& tmp_table) {
  n = tmp_table.size();
//  Log << "[node] 节点管理数据大小: " << n << endl;
  double m_avg = 0, n_avg = 0;
  for (int i = 0; i < n; i += 1) {
    m_avg += tmp_table[i][2];
    n_avg += tmp_table[i][1];
    lower = min(lower, tmp_table[i][1]);
    upper = max(upper, tmp_table[i][1]);
  }
  m_avg /= n;
  n_avg /= n;
  double cov = 0;
  for (int i = 0; i < n; i += 1) {
    cov += (tmp_table[i][2] - m_avg) * (tmp_table[i][1] - n_avg);
  }
  // cov /= n - 1;
  double var = 0;
  for (int i = 0; i < n; i += 1) {
    var += pow(tmp_table[i][2] - m_avg, 2);
  }
  // var /= n - 1;
  a = cov / var;
  b = n_avg - a * m_avg;
  e = a * (upper - lower) * error_bound / (2 * n);
//  Log << "[node] 节点关键参数: " << endl;
//  Log << "[node] 管理Key数量n=" << n << " 范围内max_key=" << upper << " 范围内min_key=" << lower << " 斜率a=" << a << " 截距b=" << b << " 信心度e=" << e << endl;
}

bool Node::is_over_lapping(pair<int, int>& P) {
  return (intersect(range(), P) != std::make_pair(0, 0));
}

pair<int, int> Node::range() {
//  Log << "[node] 节点管理数据范围: " << "(" << lower << ", " << upper << ")" << endl;
  return { lower, upper };
}

unordered_map<int, int>& Node::get_outliers() {
  return outliers;
}

void Node::set_leaf_node(bool _is_leaf_node) {
//  Log << "[trs_tree] 设置节点为叶子: " << _is_leaf_node << endl;
  is_leaf_node = _is_leaf_node;
}

bool Node::query_leaf_node() {
//  Log << "[node] " << (is_leaf_node ? "叶子节点" : "卫星节点") << endl;
  return is_leaf_node;
}

TRS_Tree::TRS_Tree() {
  root = new Node;
}

void TRS_Tree::build(vector<vector<int> >& table) {
  int len = table.size();
//  Log << "[trs_tree] 构建表原大小: " << len << endl;
  for (int i = 0; ; i += 1) {
    int to_2int = 1 << i;
    if (to_2int >= len) {
      table.reserve(to_2int);
      for (int j = len; j < to_2int; j += 1) {
        table.emplace_back(std::move(vector<int>{ table[len - 1][0], table[len - 1][1], table[len - 1][2] }));
      }
      break;
    }
  }
//  Log << "[trs_tree] 凑足后构建表大小: " << len << endl;
  queue<pair<Node*, vector<vector<int> > > > q;
  q.push({ root, table });
  while (!q.empty()) {
    auto p = q.front();
    q.pop();
    Node* node = p.first;
    vector<vector<int> >& tmp_table = p.second;
    node -> compute(tmp_table);
    if (!validate(tmp_table, node)) {
//      Log << "[trs_tree] 节点分裂" << endl;
      node -> split();
      node -> set_leaf_node(false);
      vector<vector<vector<int> > > sub_tables;
      int seg = tmp_table.size() / k;
      for (int i = 0; i < k; i += 1) {
        sub_tables.emplace_back(std::move(vector<vector<int> >(tmp_table.begin() + seg * i, tmp_table.begin() + seg * (i + 1))));
      }
      for (int i = 0; i < k; i += 1) {
        q.push(std::make_pair(node -> get_sub_node(i), sub_tables[i]));
      }
    }
  }
}

bool TRS_Tree::validate(vector<vector<int> >& tmp_table, Node* node) {
  int len = tmp_table.size();
  for (int i = 0; i < len; i += 1) {
    int key = tmp_table[i][0];
    int host = tmp_table[i][1];
    int target = tmp_table[i][2];
    auto p = node -> get_host_range(std::make_pair(target, target));
    if (host < p.first || host > p.second) {
//      Log << "[trs_tree] 插入溢出区: " << "(" << target << ", " << key << ")" << endl;
      node -> get_outliers().insert({ target, key });
    }
    if (node -> get_outliers().size() > outliers_ratio * len) {
//      Log << "[trs_tree] 溢出区满" << endl;
      return false;
    }
  }
  return true;
}

pair<set<int>, set<int> > TRS_Tree::lookup(pair<int, int>& pre) {
  set<int> range_id;
  set<int> tuple_id;
  vector<pair<int, int> > vec;
  queue<Node*> q;
  if (root -> is_over_lapping(pre)) {
    q.push(root);
  }
  while (!q.empty()) {
    Node* node = q.front();
    q.pop();
    if (node -> query_leaf_node()) {
      // 查询字段与节点node有交集，才会进入到这里
//      Log << "[trs_tree] 在叶子节点中查找" << endl;
      auto p = intersect(node -> range(), pre);
      vec.emplace_back(std::move(node -> get_host_range(p)));
      auto outlier = node -> get_outliers();
      for (int i = pre.first; i <= pre.second; i += 1) {
        if (outlier.count(i)) {
          tuple_id.insert(outlier[i]);
        }
      }
    }
    else {
//      Log << "[trs_tree] 在卫星节点中查找" << endl;
      for (int i = 0; i < k; i += 1) {
        Node* tmp_node = node -> get_sub_node(i);
        if (tmp_node -> is_over_lapping(pre)) {
          q.push(tmp_node);
        }
      }
    }
  }
  for (auto& p : vec) {
    for (int i = p.first; i <= p.second; i += 1) {
      range_id.insert(i);
    }
  }
  return { range_id, tuple_id };
}

void TRS_Tree::insert(int target, int sid, int tid) {
  Node* node = traverse(root, target);
  auto host_range = node -> get_host_range({ target, target });
  if (sid < host_range.first || sid > host_range.second) {
    node -> get_outliers().insert({ target, tid });
  }
}

void TRS_Tree::_delete(int target) {
  Node* node = traverse(root, target);
  node -> get_outliers().erase(target);
}

Node* TRS_Tree::traverse(Node* node, int target) {
  if (node -> query_leaf_node()) {
    return node;
  }
  else {
    for (int i = 0; i < k; i += 1) {
      Node* pn = node -> get_sub_node(i);
      auto p =  pn -> range();
      if (p.first <= target && target <= p.second) {
        return traverse(pn, target);
      }
    }
  }
  return nullptr;
}

long long int TRS_Tree::calMem(){
    long long int sum = 0;
    set<int> range_id;
    set<int> tuple_id;
    vector<pair<int, int> > vec;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        Node* node = q.front();
        q.pop();
        if (node -> query_leaf_node()) {    // 是叶子结点
            auto outlier = node -> get_outliers();
            sum = sum + 4 + outlier.size() * 2;
        }
        else {      // 非叶子结点
            sum = sum + 2 + k;
            for (int i = 0; i < k; i += 1) {
                Node* tmp_node = node -> get_sub_node(i);
                q.push(tmp_node);
            }
        }
    }
    return sum;
}

}