#pragma once

#ifndef MICA_ALLOC_HUGETLB_SHM_H_
#define MICA_ALLOC_HUGETLB_SHM_H_

#include <stdlib.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <libpmem.h>

namespace mica {
namespace alloc {

class HugeTLBFS_SHM {
 public:
  HugeTLBFS_SHM (int numa_node = 0, size_t memory_size = 1UL<<35);

  virtual ~HugeTLBFS_SHM();
  virtual void* malloc_contiguous(size_t size);
  virtual void  free_contiguous(void* ptr);
  virtual void* malloc_2mb ();
  virtual void  free_2mb   (void* ptr);
  virtual void  pre_malloc_2mb (void);

  virtual void  print (void) {
    printf ("memory : base_=%p size=%lu free=%lu\n", page_2mb_base_,
            memory_size_, page_2mb_free_.size());
  }

 protected:
  std::mutex mtx_;
  static constexpr size_t kPageSize_ = 2*1024 * 1024;
  int numa_node_of_shm_;
  std::unordered_map<void*,size_t> mems_;
  void *page_2mb_base_;
  size_t memory_size_;
  std::vector<void*> page_2mb_free_;

  virtual void* malloc_mem_from_system (size_t size);
  virtual void  free_mem_to_system (void *ptr, size_t size);
};

class PmemHugeTLBFS_SHM : public HugeTLBFS_SHM {
 public:
  PmemHugeTLBFS_SHM (int numa_node = 0, size_t total_size = 1UL<<37);
  ~PmemHugeTLBFS_SHM ();

  void  print (void) {
    printf ("aep-mem: base_=%p size=%lu free=%lu\n", page_2mb_base_,
            memory_size_, page_2mb_free_.size());
  }

 private:
                                               // make sure that you have proper authority of the directory
  const char* pmem_dir_ = "/mnt/mypmem";       // mounted directory
  const char* prefix_   = "test/mypmem_";   // subdirectory should exist, the last level is filename prefix 

  void* malloc_mem_from_system (size_t size);
  void  free_mem_to_system (void *ptr, size_t size);
};

}
}
#endif
