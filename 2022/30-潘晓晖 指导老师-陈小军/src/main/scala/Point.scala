package org.example

class Point(var pid: Int) {
  var index: Int = -1
  var pv: Array[Double] = Array[Double]()
  var rho: Double = Double.MinValue
  var delta: Double = Double.MaxValue
  var NNid: Int = -1
  var cid: Int = -1
  var isDP: Boolean = false

  def this(pid: Int, pv: Array[Double]) {
    this(pid)
    this.pv = pv
  }

  def this(index: Int, pid: Int, pv: Array[Double], rho: Double, delta: Double) {
    this(pid, pv)
    this.index = index
    this.rho = rho
    this.delta = delta
  }

  def this(pid: Int, pv: Array[Double], isDP: Boolean) {
    this(pid, pv)
    this.isDP = isDP
  }

  def this(pid: Int, pv: Array[Double], rho: Double, delta: Double, nnid: Int) {
    this(pid, pv)
    this.rho = rho
    this.delta = delta
    this.NNid = nnid
  }

}
