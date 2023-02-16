/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.unsafe.memory;

import javax.annotation.concurrent.GuardedBy;
import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.BitSet;
import java.util.Arrays;

import org.apache.spark.unsafe.Platform;

/**
 * Shared memory allocator for OPS.
 */
public class OpsSharedMemoryAllocator implements MemoryAllocator {

  private static final int PAGE_TABLE_SIZE = 81920;
  private MemoryBlock[] pageTable = new MemoryBlock[PAGE_TABLE_SIZE];
  private BitSet allocatedPages = new BitSet(PAGE_TABLE_SIZE);
  private List<Integer> pageNumList = new LinkedList<>();
  private long usedMemory = 0;
  private int usedPages = 0;
  private int totalPages = 0;
  // private static final long MEMORY_LIMIT = 12000;
  private long MEMORY_LIMIT = 4000000000L;

  public OpsSharedMemoryAllocator(long limit) {
    // this.MEMORY_LIMIT = limit;
    System.out.println("OpsSharedMemoryAllocator memory limit: " + this.MEMORY_LIMIT);
  }

  @GuardedBy("this")
  private final Map<Long, LinkedList<WeakReference<long[]>>> bufferPoolsBySize = new HashMap<>();

  private static final int POOLING_THRESHOLD_BYTES = 128 * 1024;

  /**
   * Returns true if allocations of the given size should go through the pooling mechanism and
   * false otherwise.
   */
  private boolean shouldPool(long size) {
    // Very small allocations are less likely to benefit from pooling.
    return size >= POOLING_THRESHOLD_BYTES;
  }

  @Override
  public MemoryBlock allocate(long size) throws OutOfMemoryError {
    if (size + this.usedMemory > MEMORY_LIMIT) {
      System.out.println("Out of memory!");
      return null;
    }
    int numWords = (int) ((size + 7) / 8);
    long alignedSize = numWords * 8L;
    assert (alignedSize >= size);
    if (shouldPool(alignedSize)) {
      synchronized (this) {
        final LinkedList<WeakReference<long[]>> pool = bufferPoolsBySize.get(alignedSize);
        if (pool != null) {
          while (!pool.isEmpty()) {
            final WeakReference<long[]> arrayReference = pool.pop();
            final long[] array = arrayReference.get();
            if (array != null) {
              assert (array.length * 8L >= size);
              MemoryBlock memory = new MemoryBlock(array, Platform.LONG_ARRAY_OFFSET, size);
              if (MemoryAllocator.MEMORY_DEBUG_FILL_ENABLED) {
                memory.fill(MemoryAllocator.MEMORY_DEBUG_FILL_CLEAN_VALUE);
              }
              synchronized (this) {
                int pageNumber = this.allocatedPages.nextClearBit(0);
                if (pageNumber >= PAGE_TABLE_SIZE) {
                  // releaseExecutionMemory(acquired, consumer);
                  throw new IllegalStateException(
                    "Have already allocated a maximum of " + PAGE_TABLE_SIZE + " pages");
                }
                this.allocatedPages.set(pageNumber);
                memory.pageNumber = pageNumber;
                this.pageTable[pageNumber] = memory;

                this.usedMemory += size;
                this.totalPages += 1;
                this.usedPages += 1;
              }
              // System.out.println(System.currentTimeMillis()/1000 + ": Allocate page " + memory.pageNumber + ", size: " + size + ", used: " + this.usedMemory/1000000 + ", " + this.usedPages + ", " + this.totalPages);
              return memory;
            }
          }
          bufferPoolsBySize.remove(alignedSize);
        }
      }
    }
    long[] array = new long[numWords];
    MemoryBlock memory = new MemoryBlock(array, Platform.LONG_ARRAY_OFFSET, size);
    if (MemoryAllocator.MEMORY_DEBUG_FILL_ENABLED) {
      memory.fill(MemoryAllocator.MEMORY_DEBUG_FILL_CLEAN_VALUE);
    }
    synchronized (this) {
      int pageNumber = this.allocatedPages.nextClearBit(0);
      if (pageNumber >= PAGE_TABLE_SIZE) {
        // releaseExecutionMemory(acquired, consumer);
        throw new IllegalStateException(
          "Have already allocated a maximum of " + PAGE_TABLE_SIZE + " pages");
      }
      this.allocatedPages.set(pageNumber);
      memory.pageNumber = pageNumber;
      this.pageTable[pageNumber] = memory;

      this.usedMemory += size;
      this.totalPages += 1;
      this.usedPages += 1;
    }
    // System.out.println(System.currentTimeMillis()/1000 + ": Allocate page " + memory.pageNumber + ", size: " + size + ", used: " + this.usedMemory/1000000 + ", " + this.usedPages + ", " + this.totalPages);
    return memory;
  }

  public void free(int pageNum) {
    this.free(this.pageTable[pageNum]);
  }

  @Override
  public void free(MemoryBlock memory) {
    assert (memory.obj != null) :
      "baseObject was null; are you trying to use the on-heap allocator to free off-heap memory?";
    assert (memory.pageNumber != MemoryBlock.FREED_IN_ALLOCATOR_PAGE_NUMBER) :
      "page has already been freed";

    // System.out.println(System.currentTimeMillis()/1000 + ": Free page " + memory.pageNumber + ", used: " + this.usedMemory/1000000 + ", " + this.usedPages + ", " + this.totalPages);
    
    final long size = memory.size();
    synchronized (this) {
      this.pageTable[memory.pageNumber] = null;
      allocatedPages.clear(memory.pageNumber);

      this.usedMemory -= size;
      this.usedPages -= 1;
    }
    memory.pointers = null;

    if (MemoryAllocator.MEMORY_DEBUG_FILL_ENABLED) {
      memory.fill(MemoryAllocator.MEMORY_DEBUG_FILL_FREED_VALUE);
    }

    // Mark the page as freed (so we can detect double-frees).
    memory.pageNumber = MemoryBlock.FREED_IN_ALLOCATOR_PAGE_NUMBER;

    // As an additional layer of defense against use-after-free bugs, we mutate the
    // MemoryBlock to null out its reference to the long[] array.
    long[] array = (long[]) memory.obj;
    memory.setObjAndOffset(null, 0);

    long alignedSize = ((size + 7) / 8) * 8;
    // System.out.println("Free page without pool, used: " + this.usedMemory);
    if (shouldPool(alignedSize)) {
      synchronized (this) {
        LinkedList<WeakReference<long[]>> pool = bufferPoolsBySize.get(alignedSize);
        if (pool == null) {
          pool = new LinkedList<>();
          bufferPoolsBySize.put(alignedSize, pool);
        }
        pool.add(new WeakReference<>(array));
      }
    } else {
      // Do nothing
    }
  }

  @Override
  public void cleanUpAllMemory() {
    this.pageTable = null;
    this.allocatedPages = null;
    this.pageNumList = null;
    this.usedMemory = 0;
    this.usedPages = 0;
    this.totalPages = 0;
  }

  @Override
  public void addSharedPage(int pageNumber) {
    synchronized(this) {
      this.pageNumList.add(pageNumber);
    }
  }

  @Override
  public List<MemoryBlock> getSharedPages() {
    synchronized(this) {
      if (this.pageNumList.size() == 0) {
        return new LinkedList<>();
      }
      List<MemoryBlock> pages = new LinkedList<>();
      for (int index : this.pageNumList) {
        pages.add(this.pageTable[index]);
      }
      this.pageNumList.clear();
      return pages;
    }
  }
}
