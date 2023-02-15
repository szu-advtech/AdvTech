#ifndef _BPTREE_H_
#define _BPTREE_H_

#include <vector>
#include "shrinkingcone_segmentation.h"
#include <algorithm>
#include <iostream>
#include "config.h"
#include <math.h>
using namespace std;

typedef long long ll;
int Se;

// BP node
class Node {
  bool IS_LEAF;
  ll *key;
  int size;
  Node **ptr;
  Segment **seg;
  friend class BPTree;

public:
  Node();
  ~Node();
};

// BP tree
class BPTree {
  Node *root;
  void insertInternal(ll, Node *, Node *);
  Node *findParent(Node *, Node *);

public:
  BPTree();
  State search(ll);
  void insert(ll);
  State construct(const vector<ll>& underdata);
  void display(Node *cursor);
//    void display_seg();
  State delta_insert(ll);
  int calculate_size();
  int internal_calculate_size(Node*);
  ~BPTree();
  void internal_destruct(Node* &);
  void destruct();
};

Node::Node() {
  size = 0;
  key = new ll[config::FANOUT];
  ptr = new Node* [config::FANOUT + 1];
  seg = new Segment* [config::FANOUT + 1];
  for (int i = 0; i < config::FANOUT + 1; i += 1) {
    ptr[i] = NULL;
    seg[i] = NULL;
  }
  seg[0] = new Segment;
}

Node::~Node() {
  delete key;

  for (int i = 0; i < size; i += 1) {
    if (ptr[i] != NULL) {
      delete ptr[i]; 
      ptr[i] = NULL;
    }
    if (seg[i] != NULL) {
      delete seg[i]; 
      seg[i] = NULL;
    }
  }
  // 这一步不必要也不能加，因为在该步处理之前， 就已经达到该步的效果了
  // 不可重复delete
//  if (size > 0) {
//    if (ptr[size] != NULL)  delete ptr[size]; ptr[size] = NULL;
//    if (seg[size] != NULL)  delete seg[size]; ptr[size] = NULL;
//  }

  if (ptr != NULL) {
    delete [] ptr; 
    ptr = NULL;
  }
  if (seg != NULL) {
    delete [] seg; 
    seg = NULL;
  }
}

BPTree::BPTree() {
  root = NULL;
}

void BPTree::internal_destruct(Node* &node) {
  if (node == NULL) {
    return;
  }
  if (node -> IS_LEAF != true) {
    for (int i = node->size; i >= 0; i -= 1) {
      internal_destruct(node -> ptr[i]);
    }
  }
  delete node;
  node = NULL;
}

BPTree::~BPTree() {
  destruct();
}

void BPTree::destruct() {
//  display(root);
  internal_destruct(root);
  root = NULL;
}

// Search operation
State BPTree::search(ll x) {
  if (root == NULL) {
    cout << "Tree is empty\n";
    return State::FAIL;
  } else {
    Node *cursor = root;
//      cout << "Lookup中结果: " << "\n";
    while (cursor->IS_LEAF == false) {
      for (int i = 0; i < cursor->size; i++) {
        if (x < cursor->key[i]) {
          cursor = cursor->ptr[i];
          break;
        }
        if (i == cursor->size - 1) {
          cursor = cursor->ptr[i + 1];
          break;
        }
      }
    }
    // 进入到叶子节点中，使用Segment算法
    if (x < cursor->key[0]) {
      if (cursor->seg[0]->search_buffer(x) == x) {
        return State::SUCCESS;
      }
      return State::FAIL;
    }
    int i = 0;
    for (; i < cursor->size; i++) {
      if (i == cursor->size - 1 || (x >= cursor->key[i] && x < cursor->key[i + 1])) {
        break;
      }
    }
    Segment *seg = cursor->seg[i + 1];
    int pos = seg->slope * (x - seg->start);
    int l_bound = max_double(pos - config::ERROR, 0);
    int r_bound = min_double(pos + config::ERROR, seg->data.size());
    pos = distance(seg->data.begin(), lower_bound(seg->data.begin() + l_bound, seg->data.begin() + r_bound, x));
    if (seg->data[pos] == x) {
      //cout << x << endl;
      return State::SUCCESS;
    }
    if (seg->search_buffer(x) == x) {
      return State::SUCCESS;
    }
  }
  return State::FAIL;
}

// Insert Operation
void BPTree::insert(ll x) {
  if (root == NULL) {
    root = new Node;
    root->key[0] = x;
    root->IS_LEAF = true;
    root->size = 1;
  } else {
    Node *cursor = root;
    Node *parent;
    while (cursor->IS_LEAF == false) {
      parent = cursor;
      for (int i = 0; i < cursor->size; i++) {
        if (x < cursor->key[i]) {
          cursor = cursor->ptr[i];
          break;
        }
        if (i == cursor->size - 1) {
          cursor = cursor->ptr[i + 1];
          break;
        }
      }
    }
    if (cursor->size < config::FANOUT) {
      int i = 0;
      while (x > cursor->key[i] && i < cursor->size)
        i++;
      for (int j = cursor->size; j > i; j--) {
        cursor->key[j] = cursor->key[j - 1];
      }
      cursor->key[i] = x;
      cursor->size++;
      cursor->ptr[cursor->size] = cursor->ptr[cursor->size - 1];
      cursor->ptr[cursor->size - 1] = NULL;
    } else {
      Node *newLeaf = new Node;
      ll virtualNode[config::FANOUT + 2];
      for (int i = 0; i < config::FANOUT; i++) {
        virtualNode[i] = cursor->key[i];
      }
      int i = 0, j;
      while (x > virtualNode[i] && i < config::FANOUT)
        i++;
      for (int j = config::FANOUT + 1; j > i; j--) {
        virtualNode[j] = virtualNode[j - 1];
      }
      virtualNode[i] = x;
      newLeaf->IS_LEAF = true;
      cursor->size = (config::FANOUT + 1) / 2;
      newLeaf->size = config::FANOUT + 1 - (config::FANOUT + 1) / 2;
      cursor->ptr[cursor->size] = newLeaf;
      newLeaf->ptr[newLeaf->size] = cursor->ptr[config::FANOUT];
      cursor->ptr[config::FANOUT] = NULL;
      for (i = 0; i < cursor->size; i++) {
        cursor->key[i] = virtualNode[i];
      }
      for (i = 0, j = cursor->size; i < newLeaf->size; i++, j++) {
        newLeaf->key[i] = virtualNode[j];
      }
      if (cursor == root) {
        Node *newRoot = new Node;
        newRoot->key[0] = newLeaf->key[0];
        newRoot->ptr[0] = cursor;
        newRoot->ptr[1] = newLeaf;
        newRoot->IS_LEAF = false;
        newRoot->size = 1;
        root = newRoot;
      } else {
        insertInternal(newLeaf->key[0], parent, newLeaf);
      }
    }
  }
}

// Insert Operation
void BPTree::insertInternal(ll x, Node *cursor, Node *child) {
  if (cursor->size < config::FANOUT) {
    int i = 0;
    while (x > cursor->key[i] && i < cursor->size)
      i++;
    for (int j = cursor->size; j > i; j--) {
      cursor->key[j] = cursor->key[j - 1];
    }
    for (int j = cursor->size + 1; j > i + 1; j--) {
      cursor->ptr[j] = cursor->ptr[j - 1];
    }
    cursor->key[i] = x;
    cursor->size++;
    cursor->ptr[i + 1] = child;
  } else {
    Node *newInternal = new Node;
    ll virtualKey[config::FANOUT + 1];
    Node *virtualPtr[config::FANOUT + 2];
    for (int i = 0; i < config::FANOUT; i++) {
      virtualKey[i] = cursor->key[i];
    }
    for (int i = 0; i < config::FANOUT + 1; i++) {
      virtualPtr[i] = cursor->ptr[i];
    }
    int i = 0, j;
    while (x > virtualKey[i] && i < config::FANOUT)
      i++;
    for (int j = config::FANOUT + 1; j > i; j--) {
      virtualKey[j] = virtualKey[j - 1];
    }
    virtualKey[i] = x;
    for (int j = config::FANOUT + 2; j > i + 1; j--) {
      virtualPtr[j] = virtualPtr[j - 1];
    }
    virtualPtr[i + 1] = child;
    newInternal->IS_LEAF = false;
    cursor->size = (config::FANOUT + 1) / 2;
    newInternal->size = config::FANOUT - (config::FANOUT + 1) / 2;
    for (i = 0, j = cursor->size + 1; i < newInternal->size; i++, j++) {
      newInternal->key[i] = virtualKey[j];
    }
    for (i = 0, j = cursor->size + 1; i < newInternal->size + 1; i++, j++) {
      newInternal->ptr[i] = virtualPtr[j];
    }
    if (cursor == root) {
      Node *newRoot = new Node;
      newRoot->key[0] = cursor->key[cursor->size];
      newRoot->ptr[0] = cursor;
      newRoot->ptr[1] = newInternal;
      newRoot->IS_LEAF = false;
      newRoot->size = 1;
      root = newRoot;
    } else {
      insertInternal(cursor->key[cursor->size], findParent(root, cursor), newInternal);
    }
  }
}

// Find the parent
Node *BPTree::findParent(Node *cursor, Node *child) {
  Node *parent;
  if (cursor->IS_LEAF || (cursor->ptr[0])->IS_LEAF) {
    return NULL;
  }
  for (int i = 0; i < cursor->size + 1; i++) {
    if (cursor->ptr[i] == child) {
      parent = cursor;
      return parent;
    } else {
      parent = findParent(cursor->ptr[i], child);
      if (parent != NULL)
        return parent;
    }
  }
  return parent;
}

State BPTree::construct(const vector<ll>& underdata) {
  try {
    vector<ll> _;
    _.resize(0);
    vector<ll> underlying_data(underdata.begin(), underdata.end());
    destruct();

    vector<Segment> underlying_segs = shrinkingcore_segmentation(underlying_data, _);
    Se = underlying_segs.size();
    for (Segment seg : underlying_segs) {
      insert(seg.start);
    }
    // display(root);
    for (int j = 0; j < Se; j += 1) {
      Node *cursor = root;
      while (cursor->IS_LEAF == false) {
        for (int i = 0; i < cursor->size; i++) {
          if (underlying_segs[j].start < cursor->key[i]) {
            cursor = cursor->ptr[i];
            break;
          }
          if (i == cursor->size - 1) {
            cursor = cursor->ptr[i + 1];
            break;
          }
        }
      }
      // 抵达叶子节点, 找到合适的指针, 使其指向下层的'线段节点'
      for (int i = 0; i < cursor->size; i++) {
        if (cursor->key[i] == underlying_segs[j].start) {
          cursor->seg[i + 1] = new Segment(underlying_segs[j]);
          break;
        }
      }
    }
    // cout << "FITing-Tree内部节点: " << "\n";
    // display(root);
    return State::SUCCESS;
  } catch (exception& e) {
    cout << e.what() << "\n";
    return State::FAIL;
  }
}

// Print the tree
void BPTree::display(Node *cursor) {
  if (cursor != NULL) {
    for (int i = 0; i < cursor->size; i++) {
      cout << cursor->key[i] << " ";
    }
    cout << "\n";
    if (cursor->IS_LEAF != true) {
      for (int i = 0; i < cursor->size + 1; i++) {
        display(cursor->ptr[i]);
      }
    }
  }
}

//void BPTree::display_seg() {
//  for (int j = 0; j < root->size; j += 1) {
//    cout << "seg: " << root->seg[j + 1]->start << " " << root->seg[j + 1]->slope << "\n";
//  }
//}

State BPTree::delta_insert(ll x) {
  try {
    Node *cursor = root;
    while (cursor->IS_LEAF == false) {
      for (int i = 0; i < cursor->size; i++) {
        if (x < cursor->key[i]) {
          cursor = cursor->ptr[i];
          break;
        }
        if (i == cursor->size - 1) {
          cursor = cursor->ptr[i + 1];
          break;
        }
      }
    }

    Segment *seg;
    // 进入到叶子节点中，使用Segment算法
    if (x < cursor->key[0]) {
      seg = cursor->seg[0];
    }
    else {
      int i = 0;
      for (; i < cursor->size; i++) {
        if (i == cursor->size - 1 || (x >= cursor->key[i] && x < cursor->key[i + 1])) {
          break;
        }
      }
      seg = cursor->seg[i + 1];
    }
    seg->insert_buffer(x);
    if (seg->is_buffer_full()) {
      vector<Segment> segs = shrinkingcore_segmentation(seg->data, seg->buf);
      for (Segment seg : segs) {
        insert(seg.start);
      }
      for (int j = 0; j < segs.size(); j += 1) {
        Node *cursor = root;
        while (cursor->IS_LEAF == false) {
          for (int i = 0; i < cursor->size; i++) {
            if (segs[j].start < cursor->key[i]) {
              cursor = cursor->ptr[i];
              break;
            }
            if (i == cursor->size - 1) {
              cursor = cursor->ptr[i + 1];
              break;
            }
          }
        }
        // 抵达叶子节点, 找到合适的指针, 使其指向下层的'线段节点'
        for (int i = 0; i < cursor->size; i++) {
          if (cursor->key[i] == segs[j].start) {
            cursor->seg[i + 1] = new Segment(segs[j]);
            break;
          }
        }
      }
      delete seg;
      seg = NULL;
      return State::SUCCESS;
    }
  } catch (exception& e) {
    std::cout << e.what() << "\n";
    return State::FAIL;
  }
  return State::FAIL;
}

int BPTree::internal_calculate_size(Node* cursor) {
  int size = 0;
  if (cursor != NULL) {
    size += sizeof(*cursor);
    if (cursor->IS_LEAF != true) {
      for (int i = 0; i < cursor->size + 1; i++) {
        size += internal_calculate_size(cursor->ptr[i]);
      }
    } else {
      // Segment 索引部分仅包含两个参数：double, long long，计16个Byte
      size += 16;
    }
  }
  return size;
}

int BPTree::calculate_size() {
  return internal_calculate_size(root);
}

double get_latency() {
  return config::C * (log2(Se) / log2(config::FANOUT) + log2(config::ERROR) + log2(config::BUFFER_SIZE));
}

#endif
