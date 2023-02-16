//
// Created by Jiananyuan on 2022/10/19.
//

#ifndef _DB_IMPL_H_
#define _DB_IMPL_H_

#include "bptree.h"
#include "config.h"
#include <limits.h>
typedef long long ll;
BPTree* fiting_tree = new BPTree;

State get(ll x) {
  return fiting_tree -> search(x);
}

State insert(ll x) {
  return fiting_tree -> delta_insert(x);
}

State construct(const vector<ll>& underdata) {
  return fiting_tree->construct(underdata);
}

int getsize() {
  return fiting_tree->calculate_size();
}

const int e[] = {1000, 100, 10};

// 性能优先保障
int performance_tradeoff(double Lreq, const vector<ll>& underdata) {
  int best_e = 0;
  int min_size = INT_MAX;
  for (int ei : e) {
    config::ERROR = ei;
    construct(underdata);
    double latency = get_latency();
    if (latency < Lreq) {
      int size = fiting_tree -> calculate_size();
      if (size < min_size) {
        min_size = size;
        best_e = ei;
      }
    }
  }
  return best_e;
}

// 存储优先保障
int size_tradeoff(double Sreq, const vector<ll>& underdata) {
  int best_e = 0;
  int min_latency = INT_MAX;
  for (int ei : e) {
    config::ERROR = ei;
    construct(underdata);
    int size = fiting_tree -> calculate_size();
    if (size < Sreq) {
      double latency = get_latency();
      if (latency < min_latency) {
        min_latency = latency;
        best_e = ei;
      }
    }
  }
  return best_e;
}

int get_e(int op, double req, const vector<ll>& underdata) {
  if (op == config::PERFORMANCE_FIRST) {
    return performance_tradeoff(req, underdata);
  }
  else if (op == config::STORAGE_FIRST) {
    return size_tradeoff(req, underdata);
  }
  return -1;
}

#endif
