package stream.test
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partitioner, SparkConf, SparkContext}

import java.io.File
import scala.util.Random

object TestRSP {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StreamRSP").setMaster("local[*]")
    val sc = new SparkContext(conf)
    toStreamRSP(10, "Items_10_5", sc)
  }

  def toStreamRSP(num: Int, path: String, sc: SparkContext): Unit = {
    val files: Array[File] = new File(path).listFiles()
    val paths: Array[String] = files.map(file => file.getPath)

    val RDDArray = new Array[RDD[(Int, (Int, String))]](num)
    val partitioner: Partitioner = new Partitioner {
      override def numPartitions = num

      override def getPartition(key: Any) = key.asInstanceOf[Int]
    }
    var i = 0
    for (subPath <- paths) {
      if (!subPath.contains("SUCCESS")){
        val mRDD: RDD[(Int, (Int, String))] = sc.textFile(subPath).map(row => (new Random().nextInt(num), (i, row)))
        RDDArray(i) = mRDD
        i = i + 1;
      }
    }
    val unionRDD: RDD[(Int, (Int, String))] = RDDArray.reduce((rdd1, rdd2) => rdd1.union(rdd2))
    val streamRSP: RDD[String] = unionRDD.partitionBy(partitioner).map(_._2).mapPartitions(f => {
      val partitionData: Array[(Int, String)] = f.toArray
      partitionData.sortBy(_._1).iterator
    }).map(_._2)
    paths.foreach(println)
    println("文件个数:" + RDDArray.length)
    println("RDD合并后分区数:" + unionRDD.getNumPartitions)
    println("StreamRSP分块数:" + streamRSP.getNumPartitions)
    streamRSP.saveAsTextFile(path + "_SRSP_" + num + "Blocks")
  }

  def toRSP(num: Int, path: String, sc: SparkContext): Unit = {
    val partitioner: Partitioner = new Partitioner {
      override def numPartitions = num
      override def getPartition(key: Any) = key.asInstanceOf[Int]
    }
    val RSPRDD: RDD[String] = sc
      .textFile(path)
      .map(row => (new Random().nextInt(num), row))
      .partitionBy(partitioner)
      .map(_._2)
    RSPRDD.saveAsTextFile(path + "_RSP_" + num + "Blocks")
  }

}
