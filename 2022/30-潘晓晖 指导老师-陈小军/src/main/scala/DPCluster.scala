package org.example

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class DPCluster(var densityType: Int, var dc: Double) {

  // compute density of each point using gaussian kernel
  private def gaussian_density(distMatrix: Array[Array[Double]], size: Int): Array[Double] = {
    val density = new Array[Double](size)
    if (size == 1) {
      density(0) = Double.MinValue
    } else {
      for (i <- 0 until size - 1) {
        for (j <- i + 1 until size) {
          val tmp = Math.exp(-(distMatrix(i)(j) / dc) * (distMatrix(i)(j) / dc))
          density(i) += tmp
          density(j) += tmp
        }
      }
    }
    density
  }

  // compute density of each point using cutoff kernel
  private def cutoff_density(distMatrix: Array[Array[Double]], size: Int): Array[Double] = {
    val density = new Array[Double](size)
    if (size == 1) {
      density(0) = 0
    } else {
      for (i <- 0 until size - 1) {
        for (j <- i + 1 until size) {
          if (distMatrix(i)(j) < dc) {
            density(i) += 1
            density(j) += 1
          }
        }
      }
    }
    density
  }

  def computeDistMatrixAndRho(points: Array[Point]): Unit = {
    // compute distance matrix
    val size = points.length
    var distMatrix = Array.ofDim[Double](size, size)
    for (i <- 0 until size - 1) {
      distMatrix(i)(i) = 0
      for (j <- i + 1 until size) {
        val dist = Util.distance(points(i).pv, points(j).pv)
        distMatrix(i)(j) = dist
        distMatrix(j)(i) = dist
      }
    }
    distMatrix(size - 1)(size - 1) = 0

    // compute density of each point
    var density: Array[Double] = null
    if (densityType == 1) {
      density = cutoff_density(distMatrix, size)
    } else {
      density = gaussian_density(distMatrix, size)
    }

    for (i <- 0 until size) {
      points(i).rho = density(i)
    }

    distMatrix = null
  }

  def computeDistMatrixAndDelta(points: Array[Point]): Unit = {
    // compute distance matrix
    val size = points.length
    var distMatrix = Array.ofDim[Double](size, size)
    for (i <- 0 until size - 1) {
      distMatrix(i)(i) = 0
      for (j <- i + 1 until size) {
        val dist = Util.distance(points(i).pv, points(j).pv)
        distMatrix(i)(j) = dist
        distMatrix(j)(i) = dist
      }
    }
    distMatrix(size - 1)(size - 1) = 0

    // sort the points by density (rho)
    val sortedPoints = mutable.PriorityQueue[Point]()(Ordering.by[Point, Double](_.rho))
    for (i <- points.indices) {
      sortedPoints.enqueue(points(i))
    }
    // get the point with maximum density
    var maxEntry = sortedPoints.dequeue()
    maxEntry.NNid = maxEntry.pid

    val highDensPoints = new ArrayBuffer[Int]()
    highDensPoints.append(maxEntry.index)
    val sameRhoCache = new ArrayBuffer[Point]()
    var curRho = maxEntry.rho

    // compute the delta of each points
    while (sortedPoints.nonEmpty) {
      maxEntry = sortedPoints.dequeue()
      if (maxEntry.rho != curRho) {
        // set the nnid of the cached
        for (p <- sameRhoCache) {
          if (highDensPoints.length == 1) {
            p.NNid = p.pid
          } else {
            for (higher <- highDensPoints) {
              if (distMatrix(p.index)(higher) < p.delta) {
                p.delta = distMatrix(p.index)(higher)
                p.NNid = points(higher).pid
              }
            }
          }
        }
        // add the points with same rho together
        for (p <- sameRhoCache) {
          highDensPoints.append(p.index)
        }
        sameRhoCache.clear()
      }
      sameRhoCache.append(maxEntry)
      curRho = maxEntry.rho
    }
    // set the nnid of the cached
    for (p <- sameRhoCache) {
      for (higher <- highDensPoints) {
        if (distMatrix(p.index)(higher) < p.delta) {
          p.delta = distMatrix(p.index)(higher)
          p.NNid = points(higher).pid
        }
      }
    }
    distMatrix = null
  }

  def rehash(rhoOrDelta: Int, points: Array[Point], dim: Int, M: Int, pi: Int,
             w: Double, A: Double, block_size_limit: Int): Unit = {
    val rehashKeys = new ArrayBuffer[String]
    val computeKeys = new ArrayBuffer[String]
    val bucket = new mutable.HashMap[String, ArrayBuffer[Point]]

    val a = Util.generate_random_a(dim, M, pi)
    val b = Util.generate_random_b(dim, M, pi)

    for (i <- 0 until M) {
      val tid = i + ":"
      for (p <- points) {
        val bidb = new StringBuffer(tid)
        for (j <- 0 until pi) {
          val id = Util.hash(p.pv, a(i)(j), b(i)(j), w)
          bidb.append(id)
          if (j < pi - 1) {
            bidb.append(":")
          }
        }

        val bid = bidb.toString
        if (bucket.contains(bid)) {
          bucket(bid).append(p)
        } else {
          val bv2 = new ArrayBuffer[Point]
          bv2.append(p)
          bucket.put(bid, bv2)
        }
      }
    }

    for (entry <- bucket) {
      val key = entry._1
      val ps = entry._2
      if (ps.size >= block_size_limit) {
        rehashKeys.append(key)
      } else {
        computeKeys.append(key)
      }
    }

    // rehash the big size block
    for (rehashKey <- rehashKeys) {
      val ps = bucket(rehashKey).toArray
      val damping = Math.sqrt(block_size_limit.toDouble / ps.length)
      val subA = Util.getSubA(A, M, pi)
      val mpi = Util.getPIbasedMw(2, w * damping, subA)
      rehash(rhoOrDelta, ps, dim, mpi._1, mpi._2, mpi._3, subA, block_size_limit)
    }

    // update distance matrix for the normal size block
    for (computeKey <- computeKeys) {
      val ps = bucket(computeKey).toArray
      val dpCluster = new DPCluster(densityType, dc)
      if (rhoOrDelta == 1) {
        dpCluster.computeDistMatrixAndRho(ps)
      } else if (rhoOrDelta == 2) {
        dpCluster.computeDistMatrixAndDelta(ps)
      } else {
        dpCluster.computeDistMatrixAndRho(ps)
        dpCluster.computeDistMatrixAndDelta(ps)
      }
    }

  }

}
