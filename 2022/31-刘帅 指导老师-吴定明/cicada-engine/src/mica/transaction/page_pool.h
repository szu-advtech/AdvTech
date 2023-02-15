#pragma once
#ifndef MICA_TRANSACTION_PAGE_POOL_H_
#define MICA_TRANSACTION_PAGE_POOL_H_

#include <cstdio>
#include <atomic>
#include "mica/util/lcore.h"

namespace mica {
namespace transaction {
template <class StaticConfig>
class PagePool {
 public:
  typedef typename StaticConfig::Alloc Alloc;
  typedef typename StaticConfig::PmemAlloc PmemAlloc;
  static constexpr uint64_t kPageSize = 2 * 1024 *1024;

  PagePool(Alloc* alloc, PmemAlloc* aep_alloc, uint8_t aep_only)
      : alloc_(alloc), aep_alloc_(aep_alloc), aep_only_(aep_only) {
    page_num_ = 0UL;
    aep_page_num_ = 0UL;
    aep_as_dram_page_num_ = 0UL;
    aep_as_dram_enabled_ = 0;
  }

  ~PagePool() {}

  char* allocate() {
    if (aep_only_) {
      char *p = static_cast<char*>(aep_alloc_->malloc_2mb ());
      aep_page_num_ ++;
      return p;
    }
    else {
      if (aep_as_dram_enabled_) {
        char *p = static_cast<char*>(aep_alloc_->malloc_2mb ());
        aep_page_num_ ++;
        aep_as_dram_page_num_ ++;
        return p;
      }
      else {
        char *p = static_cast<char*>(alloc_->malloc_2mb ());
        if (p != nullptr) {
          page_num_ ++;
          return p;
        }
        else {
          aep_as_dram_enabled_ = 1;
          aep_page_num_ ++;
          aep_as_dram_page_num_ ++;
          p = static_cast<char*>(aep_alloc_->malloc_2mb ());
        }
        return p;
      }
    }
    return nullptr;
  }

  void free(char* p) {
    if (aep_only_) {
      aep_alloc_->free_2mb (p);
      aep_page_num_ --;
    }
    else {
      alloc_->free_2mb (p);
      page_num_ --;
    }
  }

  void print_status() const {
    printf("PagePool aep_only=%" PRIu8 "\n", aep_only_);
    printf("  in use     : %7.3lf GB\n",
           static_cast<double>((page_num_+aep_page_num_) * kPageSize) /
               1000000000.);
    printf("  in use dram: %7.3lf GB\n",
           static_cast<double>((page_num_) * kPageSize) /
               1000000000.);
    printf("  in use dcpm: %7.3lf GB\n",
           static_cast<double>((aep_page_num_) * kPageSize) /
               1000000000.);
    printf("  aep-as-dram: %7.3lf GB\n",
           static_cast<double>(aep_as_dram_page_num_ * kPageSize) /
               1000000000.);
  }

  uint64_t both_page_num () { return page_num_+aep_page_num_; }
  double   both_memory_use () { 
    return static_cast<double>((page_num_+aep_page_num_)*kPageSize)/
               1000000000.0; }
  uint8_t  aep_only () { return aep_only_; }
  uint64_t transform_page_num () { return aep_as_dram_page_num_; }
  double   transform_page_size() {
     return static_cast<double>((aep_as_dram_page_num_)*kPageSize)/
               1000000000.0; }
 
 private:
  Alloc    *alloc_;
  PmemAlloc *aep_alloc_;
  uint8_t   aep_only_, aep_as_dram_enabled_;
  std::atomic<uint64_t> page_num_;
  std::atomic<uint64_t> aep_page_num_;
  std::atomic<uint64_t> aep_as_dram_page_num_;

} __attribute__((aligned(64)));

}
}

#endif
