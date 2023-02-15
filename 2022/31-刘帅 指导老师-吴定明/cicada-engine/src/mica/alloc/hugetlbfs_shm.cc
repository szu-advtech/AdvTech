// #pragma once
#ifndef MICA_ALLOC_HUGETLB_SHM_CC_
#define MICA_ALLOC_HUGETLB_SHM_CC_

#include <numa.h>
#include <assert.h>
#include "hugetlbfs_shm.h"

namespace mica {
namespace alloc {

HugeTLBFS_SHM::HugeTLBFS_SHM (int numa_node, size_t memory_size) {
  numa_node_of_shm_ = numa_node;
  memory_size_ = memory_size;
  page_2mb_base_ = nullptr;
}

void HugeTLBFS_SHM::pre_malloc_2mb () {
  page_2mb_base_ = malloc_contiguous (memory_size_);
  memset (page_2mb_base_, 0, memory_size_);

  char *pstart = (char*)page_2mb_base_;
  char *pend   = ((char*)page_2mb_base_)+memory_size_;
  for (char *p= pstart; p< pend; p+= kPageSize_) {
    page_2mb_free_.push_back(p);
  }
}

HugeTLBFS_SHM::~HugeTLBFS_SHM () {
  for (auto i=mems_.begin(); i!=mems_.end(); i++) {
    void  *p = i->first;
    size_t sz= i->second;
    free_mem_to_system (p, sz);
  }
}

void* HugeTLBFS_SHM::malloc_mem_from_system (size_t size) {
  void *p = numa_alloc_onnode (size, numa_node_of_shm_);
  return p;
}

void HugeTLBFS_SHM::free_mem_to_system (void *ptr, size_t size) {
  numa_free (ptr, size);
}

void* HugeTLBFS_SHM::malloc_contiguous (size_t size) {
  mtx_.lock ();
  void *p = malloc_mem_from_system (size);
  mems_[p] = size;
  mtx_.unlock ();
  return p;
}

void* HugeTLBFS_SHM::malloc_2mb () {
  void *p = nullptr;

  mtx_.lock ();
  if (page_2mb_free_.empty() == false) {
    p = page_2mb_free_.back();
    page_2mb_free_.pop_back ();
  }
  mtx_.unlock ();

  // assert (p!= nullptr);
  return p;
}

void HugeTLBFS_SHM::free_contiguous (void *ptr) {
  (void) ptr;
}

void HugeTLBFS_SHM::free_2mb (void *ptr) {
  mtx_.lock ();
  page_2mb_free_.push_back (ptr);
  mtx_.unlock ();
}

// PmemHugeTLBFS_SHM

PmemHugeTLBFS_SHM::PmemHugeTLBFS_SHM (int numa_node, size_t total_size)
  : HugeTLBFS_SHM(numa_node, total_size) {} 
 
PmemHugeTLBFS_SHM::~PmemHugeTLBFS_SHM () {
  for (auto i=mems_.begin(); i!=mems_.end(); i++) {
    void  *p = i->first;
    size_t sz= i->second;
    free_mem_to_system (p, sz);
  }
  mems_.clear();
}

void* PmemHugeTLBFS_SHM::malloc_mem_from_system (size_t size) {
  char path[256];
  int charnum = snprintf (path, 256, "%s%d/%s%04lu", pmem_dir_, 
                             numa_node_of_shm_, prefix_, mems_.size());

  printf ("path= %s\n", path);
  (void) charnum;
  assert (charnum < 256);

  void *pmemaddr = nullptr;
  size_t mapped_len = size;
  int is_pmem;
  if ((pmemaddr = pmem_map_file(path, size, PMEM_FILE_CREATE, 
                                0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert ((is_pmem) && (mapped_len == size));
  return pmemaddr;	
}

void PmemHugeTLBFS_SHM::free_mem_to_system (void *ptr, size_t size) {
  pmem_unmap (ptr, size);
}

}
}
#endif
