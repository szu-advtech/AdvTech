package org.example

import jsc.distributions.Normal
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.ddogleg.optimization.{FactoryOptimization, UtilOptimize}
import org.ddogleg.optimization.functions.FunctionNtoM

import scala.collection.mutable.ArrayBuffer

object DeltaCorrect {
  // define the function to be optimized
  private class InputFunctionForRho(A: Double, M: Int, pi: Int) extends FunctionNtoM {

    override def process(input: Array[Double], output: Array[Double]): Unit = {
      // find the appropriate input(0) i.e. w/dc that make output(0) close to zero
      val tmp = Math.pow(1 - A, 1.0 / M)
      val accuracy = Math.pow(1 - tmp, 1.0 / pi)
      output(0) = 1 - 4 / (Math.sqrt(2 * Math.PI) * input(0)) - accuracy
    }

    override def getNumOfInputsN: Int = 1

    override def getNumOfOutputsM: Int = 1
  }

  private class InputFunctionForDelta(A: Double, M: Int, pi: Int) extends FunctionNtoM {
    override def process(input: Array[Double], output: Array[Double]): Unit = {
      val norm = new Normal(0, 1)
      val tmp = Math.pow(1 - A, 1.0 / M)
      val accuracy = Math.pow(1 - tmp, 1.0 / pi)
      output(0) = 2 * norm.cdf(input(0)) - 1 -
        2 * (1 - Math.exp(-input(0) * input(0) / 2)) / (Math.sqrt(2 * Math.PI) * input(0)) - accuracy
    }

    override def getNumOfInputsN: Int = 1

    override def getNumOfOutputsM: Int = 1
  }

  // compute the distance of two vectors
  private def distance(p1: Array[Double], p2: Array[Double]): Double = {
    if (p1.length != p2.length) throw new Exception("vector dimension not match!!!")

    var res = 0.0
    for (i <- p1.indices) {
      res += (p1(i) - p2(i)) * (p1(i) - p2(i))
    }

    Math.sqrt(res)
  }

  // use TrustRegion Dogleg algorithm to find the optimal w for the given accuracy, L, and l
  private def get_w(A1: Double, L: Int, l: Int, dc: Double): Double = {
    val func = new InputFunctionForRho(A1, L, l)
    val optimizer = FactoryOptimization.dogleg(null, true)
    optimizer.setFunction(func, null)
    optimizer.initialize(Array[Double](10), 1e-12, 1e-12)
    UtilOptimize.process(optimizer, 500)
    val found = optimizer.getParameters
    found(0) * dc / 2
  }

  // use TrustRegion Dogleg algorithm to find the delta bound for the given accuracy, L, and l
  private def get_delta_bound(A2: Double, L: Int, l: Int, w: Double): Double = {
    val func = new InputFunctionForDelta(A2, L, l)
    val optimizer = FactoryOptimization.dogleg(null, true)
    optimizer.setFunction(func, null)
    optimizer.initialize(Array[Double](10), 1e-12, 1e-12)
    UtilOptimize.process(optimizer, 500)
    val found = optimizer.getParameters
    (w / found(0)) / 3
  }

  private def printUsage(): Unit = {
    println("<points input path> <rho_delta input path> <output path> <point dimensions> <dc>")
    println("\t-r # of reducers\n" +
      "\t-A1 accuracy requirement for rho" +
      "\t-A2 accuracy requirement for delta" +
      "\t-l # of hash functions in each band\n" +
      "\t-L # of bands\n" +
      "\t-mr the min density of high density points\n" +
      "\t-d delim\n")
  }

  def main(args: Array[String]): Unit = {
    // check and get parameters
    if (args.length < 5) {
      printUsage()
      return
    }

    var partitions: Int = 0
    var A1: Double = 0.99
    var A2: Double = 0.9
    var l: Int = 3
    var L: Int = 3
    var mr: Double = 180
    var delim: String = " "
    var pointsPath: String = "D:/panxh/Documents/lsh_test/aggregation.txt"
    var rhoAndDeltaPath: String = "D:/panxh/Documents/lsh_test/rho_delta.txt"
    var outputPath: String = "D:/panxh/Documents/lsh_test/correct_delta_output"
    var dimension: Int = 2
    var dc: Double = 0.0
    val otherArgs = new ArrayBuffer[String]()

    var index = 0
    while (index < args.length) {
      if ("-r".equals(args(index))) {
        index += 1
        partitions = args(index).toInt
      } else if ("-A1".equals(args(index))) {
        index += 1
        A1 = args(index).toDouble
      } else if ("-A2".equals(args(index))) {
        index += 1
        A2 = args(index).toDouble
      } else if ("-l".equals(args(index))) {
        index += 1
        l = args(index).toInt
      } else if ("-L".equals(args(index))) {
        index += 1
        L = args(index).toInt
      } else if ("-mr".equals(args(index))) {
        index += 1
        mr = args(index).toDouble
      } else if ("-d".equals(args(index))) {
        index += 1
        delim = args(index)
      } else {
        otherArgs += args(index)
      }
      index += 1
    }

    if (otherArgs.length < 5) {
      println("ERROR: Wrong number of parameters: " + otherArgs.length + ".")
      printUsage()
      return
    }

    try {
      pointsPath = otherArgs(0)
      rhoAndDeltaPath = otherArgs(1)
      outputPath = otherArgs(2)
      dimension = otherArgs(3).toInt
      dc = otherArgs(4).toDouble
    } catch {
      case _: NumberFormatException =>
        println("ERROR: Wrong type of parameters.")
        printUsage()
        return
    }

    // configure and get spark context
    val conf = new SparkConf().setMaster("local[2]").setAppName("correct_delta")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc = session.sparkContext
    sc.setLogLevel("ERROR")

    // use trust region dogleg algorithm to find the optimal w
    val w: Double = get_w(A1, L, l, dc)
    println("w: " + w)
    // use trust region dogleg algorithm to find the low bound of delta
    val deltaBound = get_delta_bound(A2, L, l, w)
    println("delta_bound: " + deltaBound)
    println("min_rho: " + mr)

    // load data
    println("loading data...")
    val rdValue = sc.textFile(rhoAndDeltaPath).map(line => {
      val arr = line.trim.split("#")
      val pid = arr(0).toInt
      val rho = arr(1).toDouble
      val delta = arr(2).toDouble
      val nn = arr(3).toInt
      (pid, (rho, delta, nn))
    })

    val pointsValue = sc.textFile(pointsPath).map(line => {
      val s = line.trim.split(delim)
      val pid = s(0).toInt
      val pv = s.slice(1, dimension + 1).map(_.toDouble)
      (pid, pv)
    })

    // join two RDD to form the sample high density points
    println("sample the high density points...")
    val highDensSample = pointsValue.join(rdValue.filter(_._2._1 > mr))
      .map(point => {
      val pid = point._1
      val tuple = point._2
      val pValue = tuple._1
      val rho = tuple._2._1
      val delta = tuple._2._2
      val nn = tuple._2._3
      new Point(pid, pValue, rho, delta, nn)
    }).collect()

    // filter the points should be corrected and correct the delta
    println("filtering and correct the delta...")
    val correctedRes = pointsValue.join(rdValue).map(point => {
      val pid = point._1
      val tuple = point._2
      val rho = tuple._2._1
      var minDelta = tuple._2._2
      var minNNid = tuple._2._3
      if (minDelta > deltaBound) {
        for (high <- highDensSample) {
          if (high.rho > rho) {
            val pv1 = tuple._1
            val pv2 = high.pv
            if (distance(pv1, pv2) < minDelta) {
              minDelta = distance(pv1, pv2)
              minNNid = high.pid
            }
          }
        }
      }
      pid + "#" + rho + "#" + minDelta + "#" + minNNid
    })
    println("corrected the all the delta!")

    // save the result to file
    println("saving the result...")
    correctedRes.saveAsTextFile(outputPath)

  }

}
