package org.example

import jsc.distributions.Normal

import scala.util.Random

object Util {
  // compute the distance of two vectors
  def distance(p1: Array[Double], p2: Array[Double]): Double = {
    if (p1.length != p2.length) throw new Exception("vector dimension not match!!!")

    var res = 0.0
    for (i <- p1.indices) {
      res += (p1(i) - p2(i)) * (p1(i) - p2(i))
    }

    Math.sqrt(res)
  }

  // generate the random vectors (a),
  // every element is chosen from standard gaussian distribution
  def generate_random_a(dim: Int, L: Int, l: Int): Array[Array[Array[Double]]] = {
    val res = Array.ofDim[Double](L, l, dim)
    for (i <- 0 until L) {
      for (j <- 0 until l) {
        for (k <- 0 until dim) {
          res(i)(j)(k) = Random.nextGaussian()
        }
      }
    }
    res
  }

  // generate the random numbers (b),
  // which are uniformly distributed in [0, w]
  def generate_random_b(w: Double, L: Int, l: Int): Array[Array[Double]] = {
    val res = Array.ofDim[Double](L, l)
    for (i <- 0 until L) {
      for (j <- 0 until l) {
        res(i)(j) = Random.nextDouble() * w
      }
    }
    res
  }

  // compute the LSH function value of a data point
  def hash(p: Array[Double], a: Array[Double], b: Double, w: Double): Int = {
    if (p.length != a.length) throw new Exception("vector dimension not match!!!")
    var res: Double = 0
    for (i <- p.indices) {
      res += p(i) * a(i)
    }
    ((res + b) / w).toInt
  }

  def getSubA(A: Double, M: Int, pi: Int): Double = {
    val tmp = Math.pow(1 - A, 1.0 / M)
    val subA = Math.pow(1 - tmp, 1.0 / pi)
    subA
  }

  def getPIbasedMw(Mstart: Int, w: Double, accuracy: Double): (Int, Int, Double) = {
    var pi = Integer.MIN_VALUE
    val normrand = new Normal()
    var M = Mstart - 1
    val P = 2 * normrand.cdf(w) - 1 - 4 * (1 - Math.exp(-w * w / 8)) / (Math.sqrt(2 * Math.PI) * w)

    // find pi, that is larger than PIstart
    // so we can perform more hash and divide the space into more partitions
    while (pi < 1) {
      M += 1
      pi = Math.floor(Math.log(1 - Math.pow(1 - accuracy, 1.0 / M)) / Math.log(P)).toInt
    }
    (M, pi, w)
  }
}
