package org.example

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object GetDc {

  private def printUsage(): Unit = {
    println("<input path> <output path> <sample size> <dim> <percent>")
    println("\t-d delim\n")
  }

  def main(args: Array[String]): Unit = {
    println("received parameters: " + args.mkString("Array(", ", ", ")"))
    // check and get parameters
    if (args.length < 5) {
      printUsage()
      return
    }

    var delim: String = " "
    var inputPath: String = "D:/panxh/Documents/lsh_test/aggregation.txt"
    var outputPath: String = "D:/panxh/Documents/lsh_test/dc_output"
    var sampleSize: Int = 200
    var dimension: Int = 2
    var percent: Double = 4.0
    val otherArgs = new ArrayBuffer[String]()

    var index = 0
    while (index < args.length) {
      if ("-d".equals(args(index))) {
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
      inputPath = otherArgs(0)
      outputPath = otherArgs(1)
      sampleSize = otherArgs(2).toInt
      dimension = otherArgs(3).toInt
      percent = otherArgs(4).toDouble
    } catch {
      case _: NumberFormatException =>
        println("ERROR: Wrong type of parameters.")
        printUsage()
        return
    }

    // configure and get spark context
    val conf = new SparkConf().setMaster("local[2]").setAppName("get_dc")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc = session.sparkContext
    sc.setLogLevel("ERROR")

    // load data
    val lines = sc.textFile(inputPath)

    // sample
    val samples = lines.takeSample(withReplacement = false, sampleSize)
      .map(line => {
        val s = line.trim.split(delim)
        val point_id = s(0).toInt
        val point_value = s.slice(1, dimension + 1).map(_.toDouble)
        new Point(point_id, point_value)
      })
    println("sample size: " + samples.length)

    // compute distance and dc
    val dist_size = (samples.length * samples.length - samples.length) / 2
    val distances = new Array[Double](dist_size)

    var k = 0
    for (i <- samples.indices) {
      for (j <- i + 1 until samples.length) {
        distances(k) = Util.distance(samples(i).pv, samples(j).pv)
        k += 1
      }
    }
    val sorted_distances = distances.sorted

    val position = Math.round(dist_size * percent / 100).toInt
    println("position: " + position + "/" + dist_size)
    val dc = sorted_distances(position)
    println("dc: " + dc)

    // save the result dc
    sc.makeRDD(Seq("dc#" + dc)).saveAsTextFile(outputPath)
  }

}
