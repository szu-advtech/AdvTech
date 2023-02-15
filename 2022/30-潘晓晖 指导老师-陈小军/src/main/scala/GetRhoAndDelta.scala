package org.example

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{Partitioner, SparkConf}
import org.ddogleg.optimization.{FactoryOptimization, UtilOptimize}
import org.ddogleg.optimization.functions.FunctionNtoM

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object GetRhoAndDelta {
  // define the function to be optimized
  private class InputFunction(A: Double, M: Int, pi: Int) extends FunctionNtoM {

    override def process(input: Array[Double], output: Array[Double]): Unit = {
      // find the appropriate input(0) i.e. w/dc that make output(0) close to zero
      val tmp = Math.pow(1 - A, 1.0 / M)
      val accuracy = Math.pow(1 - tmp, 1.0 / pi)
      output(0) = 1 - 4 / (Math.sqrt(2 * Math.PI) * input(0)) - accuracy
    }

    override def getNumOfInputsN: Int = 1

    override def getNumOfOutputsM: Int = 1
  }

  // define partitioner
  private class LSHPartitioner(partitionID: mutable.HashMap[String, Int]) extends Partitioner {

    override def numPartitions: Int = partitionID.size

    override def getPartition(key: Any): Int = partitionID(key.toString)
  }

  // use TrustRegion Dogleg algorithm to find the optimal w for the given accuracy, L, and l
  private def get_w(A: Double, L: Int, l: Int, dc: Double): Double = {
    val func = new InputFunction(A, L, l)
    val optimizer = FactoryOptimization.dogleg(null, true)
    optimizer.setFunction(func, null)
    optimizer.initialize(Array[Double](10), 1e-12, 1e-12)
    UtilOptimize.process(optimizer, 500)
    val found = optimizer.getParameters
    found(0) * dc / 2
  }

  private def printUsage(): Unit = {
    println("<input path> <output path> <point dimensions> <dc>")
    println("\t-r # of reducers\n" +
      "\t-A accuracy requirement\n" +
      "\t-l # of hash functions in each band\n" +
      "\t-L # of bands\n" +
      "\t-k distance type (gaussian or cutoff)\n" +
      "\t-d delim\n" +
      "\t-sl size limit (MB)\n")
  }

  def main(args: Array[String]): Unit = {
    // check and get parameters
    if (args.length < 4) {
      printUsage()
      return
    }

    var partitions: Int = 0
    var A: Double = 0.99
    var l: Int = 3
    var L: Int = 3
    var densityType: Int = 0 // 0: gaussian, 1: cutoff
    var delim: String = " "
    var sizeLimit: Int = 200 // 200MB
    var inputPath: String = "D:/panxh/Documents/lsh_test/aggregation.txt"
    var outputPath: String = "D:/panxh/Documents/lsh_test/rho_delta_output"
    var dimension: Int = 2
    var dc: Double = 0.0
    val otherArgs = new ArrayBuffer[String]()

    var index = 0
    while (index < args.length) {
      if ("-r".equals(args(index))) {
        index += 1
        partitions = args(index).toInt
      } else if ("-A".equals(args(index))) {
        index += 1
        A = args(index).toDouble
      } else if ("-l".equals(args(index))) {
        index += 1
        l = args(index).toInt
      } else if ("-L".equals(args(index))) {
        index += 1
        L = args(index).toInt
      } else if ("-k".equals(args(index))) {
        index += 1
        val typeStr = args(index)
        if (typeStr.equals("gaussian")) {
          densityType = 0
        } else if (typeStr.equals("cutoff")) {
          densityType = 1
        }
      } else if ("-d".equals(args(index))) {
        index += 1
        delim = args(index)
      } else if ("-sl".equals(args(index))) {
        index += 1
        sizeLimit = args(index).toInt
      } else {
        otherArgs += args(index)
      }
      index += 1
    }

    if (otherArgs.length < 4) {
      println("ERROR: Wrong number of parameters: " + otherArgs.length + ".")
      printUsage()
      return
    }

    try {
      inputPath = otherArgs(0)
      outputPath = otherArgs(1)
      dimension = otherArgs(2).toInt
      dc = otherArgs(3).toDouble
    } catch {
      case _: NumberFormatException =>
        println("ERROR: Wrong type of parameters.")
        printUsage()
        return
    }

    // configure and get spark context
    val conf = new SparkConf().setMaster("local[2]").setAppName("lsh_rho_and_delta")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc = session.sparkContext
    sc.setLogLevel("ERROR")

    // load data
    // "pid+delim+pValue" => (pid, point)
    val points = sc.textFile(inputPath).map(line => {
      val s = line.trim.split(delim)
      val pid = s(0).toInt
      val pv = s.slice(1, dimension + 1).map(_.toDouble)
      (pid, pv)
    })

    // use trust region dogleg algorithm to find the optimal w
    val w: Double = get_w(A, L, l, dc)
    println("w: " + w)

    // generate random parameters used by LSH function in rho step
    println("generating a and b on rho step...")
    var a1 = Util.generate_random_a(dimension, L, l)
    var b1 = Util.generate_random_b(w, L, l)
    val broadcastA1 = sc.broadcast(a1)
    val broadcastB1 = sc.broadcast(b1)
    a1 = null
    b1 = null

    // assign partition ID by applying LSH function to each point (rho step)
    // (pid, point) => ("i:h1:h2:...", (pid, point))
    println("assigning partition ID on rho step...")
    val hashedPoints: RDD[(String, (Int, Array[Double]))] = points.flatMap(point => {
      val pid = point._1
      val pValue = point._2
      val outKey = new mutable.StringBuilder
      val a = broadcastA1.value
      val b = broadcastB1.value
      val res = new ArrayBuffer[(String, (Int, Array[Double]))]

      for (i <- 0 until L) {
        outKey ++= i + ":"
        for (j <- 0 until l) {
          val hashValue = Util.hash(pValue, a(i)(j), b(i)(j), w)
          outKey ++= hashValue.toString
          if (j < l - 1) {
            outKey ++= ":"
          }
        }
        res.append((outKey.toString(), (pid, pValue)))
        outKey.clear()
      }

      res
    })

    // repartition (rho step)
    println("LSH partition on rho step: \n" + hashedPoints.countByKey())
    val keys_r = hashedPoints.keys.distinct().collect()
    var partitionID_r = new mutable.HashMap[String, Int]
    for (i <- keys_r.indices) {
      partitionID_r.put(keys_r(i), i)
    }
    val partitionIDB_r = sc.broadcast(partitionID_r)
    partitionID_r = null
    val partitionedPoints = hashedPoints.partitionBy(new LSHPartitioner(partitionIDB_r.value))

    // local computing density (rho) in every partition
    // ("i:h1:h2:...", (pid, point)) => (pid, rho)
    println("local computing rho...")
    val localRho = partitionedPoints.mapPartitions(iter => {
      // get all the local points
      val pBuffer = new ArrayBuffer[Point]
      var index: Int = 0
      while (iter.hasNext) {
        val idAndValue = iter.next()._2
        val pid = idAndValue._1
        val pointValue = idAndValue._2
        pBuffer.append(new Point(index, pid, pointValue, Double.MinValue, Double.MaxValue))
        index += 1
      }
      val localPoints = pBuffer.toArray
      pBuffer.clear()

      val block_size_limit = Math.sqrt(sizeLimit * 1000000 / 8).toInt
      val dpCluster = new DPCluster(densityType, dc)
      if (localPoints.length < block_size_limit) {
        dpCluster.computeDistMatrixAndRho(localPoints)
      } else {
        val damping = Math.sqrt(block_size_limit.toDouble / localPoints.length)
        val subA = Util.getSubA(A, L, l)
        val mpi = Util.getPIbasedMw(2, w * damping, subA)
        dpCluster.rehash(1, localPoints, dimension, mpi._1, mpi._2, mpi._3, subA, block_size_limit)
      }

      // return the local computing result
      val result = new ArrayBuffer[(Int, Double)]
      for (p <- localPoints) {
        val pid = p.pid
        val rho = p.rho
        result.append((pid, rho))
      }

      result.toArray.iterator

    })

    // aggregate local results to get the final result
    // (pid, rho) => (pid, max_rho)
    println("get the finalRho!")
    val finalRho = localRho.reduceByKey(Math.max)
//    finalRho.checkpoint()

    // generate random parameters used by LSH function in delta step
    println("generating a and b on delta step...")
    var a2 = Util.generate_random_a(dimension, L, l)
    var b2 = Util.generate_random_b(w, L, l)
    val broadcastA2 = sc.broadcast(a2)
    val broadcastB2 = sc.broadcast(b2)
    a2 = null
    b2 = null

    // join tow RDD to form the input (pid, (maxRho, point))
    println("joining tow RDD to form (pid, (maxRho, point)...")
    val pointsWithRho = finalRho.join(points)

    // assign partition ID by applying LSH function to each point
    // (pid, (maxRho, point)) => ("i:h1:h2:...", (pid, maxRho, point))
    println("assign partition ID on delta step...")
    val hashedPointsWithRho = pointsWithRho.flatMap(point => {
      val pid = point._1
      val rhoAndValue = point._2
      val maxRho = rhoAndValue._1
      val pValue = rhoAndValue._2
      val outKey = new mutable.StringBuilder
      val a = broadcastA2.value
      val b = broadcastB2.value
      val res = new ArrayBuffer[(String, (Int, Double, Array[Double]))]

      for (i <- 0 until L) {
        outKey ++= i + ":"
        for (j <- 0 until l) {
          val hashValue = Util.hash(pValue, a(i)(j), b(i)(j), w)
          outKey ++= hashValue.toString
          if (j < l - 1) {
            outKey ++= ":"
          }
        }
        res.append((outKey.toString(), (pid, maxRho, pValue)))
        outKey.clear()
      }

      res
    })

    // repartition (delta step)
    println("LSH partition on delta step: \n" + hashedPointsWithRho.countByKey())
    val keys_d = hashedPointsWithRho.keys.distinct().collect()
    var partitionID_d = new mutable.HashMap[String, Int]
    for (i <- keys_d.indices) {
      partitionID_d.put(keys_d(i), i)
    }
    val partitionIDB_d = sc.broadcast(partitionID_d)
    partitionID_d = null
    val partitionedPointsWithRho = hashedPointsWithRho.partitionBy(new LSHPartitioner(partitionIDB_d.value))

    // local computing minimum distance (delta) in every partition
    // ("i:h1:h2:...", (pid, rho, point)) => (pid, (rho, delta, NN))
    println("local computing delta...")
    val localDelta = partitionedPointsWithRho.mapPartitions(iter => {
      // get all the local points
      val pBuffer = new ArrayBuffer[Point]
      var index: Int = 0
      while (iter.hasNext) {
        val tuple = iter.next()._2
        val pid = tuple._1
        val rho = tuple._2
        val pointValue = tuple._3
        pBuffer.append(new Point(index, pid, pointValue, rho, Double.MaxValue))
        index = index + 1
      }
      val localPoints = pBuffer.toArray
      pBuffer.clear()

      val block_size_limit = Math.sqrt(sizeLimit * 1000000 / 8).toInt
      val dpCluster = new DPCluster(densityType, dc)
      if (localPoints.length < block_size_limit) {
        dpCluster.computeDistMatrixAndDelta(localPoints)
      } else {
        val damping = Math.sqrt(block_size_limit.toDouble / localPoints.length)
        val subA = Util.getSubA(A, L, l)
        val mpi = Util.getPIbasedMw(2, w * damping, subA)
        dpCluster.rehash(2, localPoints, dimension, mpi._1, mpi._2, mpi._3, subA, block_size_limit)
      }

      // return the local computing result
      val result = new ArrayBuffer[(Int, (Double, Double, Int))]
      for (p <- localPoints) {
        val pid = p.pid
        val rho = p.rho
        val delta = p.delta
        val NN = p.NNid
        result.append((pid, (rho, delta, NN)))
      }

      result.toArray.iterator

    })

    // aggregate local results to get the final result
    // (pid, (rho, delta, NN)) => (pid, (rho, min_delta, min_NN))
    println("get the final delta!")
    val finalDelta = localDelta.reduceByKey((d1, d2) => {
      var res = d1
      if (d1._2 > d2._2) {
        res = d2
      }
      res
    })
//    finalDelta.checkpoint()

    println("saving the result...")
    finalDelta.map(item => {
      val pid = item._1
      val tuple = item._2
      val rho = tuple._1
      val delta = tuple._2
      val NN = tuple._3
      pid + "#" + rho + "#" + delta + "#" + NN
    }).saveAsTextFile(outputPath)
  }

}
