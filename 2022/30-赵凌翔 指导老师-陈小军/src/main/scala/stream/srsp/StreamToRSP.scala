package stream.srsp

import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{Partitioner, SparkConf, SparkContext}

import scala.util.Random


object StreamToRSP {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StreamRSP").setMaster("yarn")
    val sc = new SparkContext(conf)
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val files = scala.collection.mutable.ListBuffer.empty[String]
    val fs = FileSystem.get(sc.hadoopConfiguration)
    //val status: Array[FileStatus] = fs.listStatus(new Path("/user/zhaolingxiang/Covertype.parquet"))
    val status: Array[FileStatus] = fs.listStatus(new Path("/user/zhaolingxiang/HIGGS.parquet"))
    status.foreach(x=> files += x.getPath.toString)
    val paths: Array[String] = files.toArray
    toStreamRSP(paths.length - 1, paths, spark)
  }

  def toStreamRSP(num: Int, paths: Array[String], spark: SparkSession): Unit = {
    println("num: " + num)
    var i = 0
    val array = new Array[RDD[(Int, (Int, Row))]](num)
    val partitioner: Partitioner = new Partitioner {
      override def numPartitions = num
      override def getPartition(key: Any) = key.asInstanceOf[Int]
    }
    for (path <- paths) {
      if (!path.contains("SUCCESS")) {
        println("path:" + path)
        val mRDD: RDD[(Int, (Int, Row))] = spark.read.parquet(path).rdd.map(row => (new Random().nextInt(num), (i, row)))
        array(i) = mRDD
        i = i + 1
      }
    }
    val value: RDD[(Int, (Int, Row))] = array.reduce((rdd1, rdd2) => rdd1.union(rdd2))
    val fin: RDD[Row] = value.repartitionAndSortWithinPartitions(partitioner).map(_._2).mapPartitions(f => {
      val array1: Array[(Int, Row)] = f.toArray
      array1.sortBy(_._1).iterator
    }).map(_._2)

    val kvv: RDD[(Double, DenseVector)] = fin.map(row => (row.getDouble(0), row.get(1).asInstanceOf[DenseVector]))
    import spark.implicits._
    //kvv.toDF("label", "features").write.parquet("Covertype_RSP2.parquet")
    kvv.toDF("label", "features").write.parquet("HIGGS_RSP.parquet")
  }

//  def toStreamRSP(num: Int, paths: Array[String], spark: SparkSession): Unit = {
//    println("num: " + num)
//    var i = 0
//    val array = new Array[RDD[(Int, (Int, Row))]](num)
//    val partitioner: Partitioner = new Partitioner {
//      override def numPartitions = num
//
//      override def getPartition(key: Any) = key.asInstanceOf[Int]
//    }
//    for (path <- paths) {
//      if (!path.contains("SUCCESS")) {
//        println("path:" + path)
//        val mRDD: RDD[(Int, (Int, Row))] = spark.read.parquet(path).rdd.repartition(1).map(row => (new Random().nextInt(num), row)).repartitionAndSortWithinPartitions(partitioner).map(row => (row._1, (i, row._2)))
//        array(i) = mRDD
//        i = i + 1
//      }
//    }
//    val value: RDD[(Int, (Int, Row))] = array.reduce((rdd1, rdd2) => rdd1.union(rdd2))
//    println("=====================value:" + value.getNumPartitions)
//    val fin: RDD[Row] = value.repartitionAndSortWithinPartitions(partitioner).map(_._2).mapPartitions(f => {
//      val array1: Array[(Int, Row)] = f.toArray
//      array1.sortBy(_._1).iterator
//    }).map(_._2)
//
//    val kvv: RDD[(Double, DenseVector)] = fin.map(row => (row.getDouble(0), row.get(1).asInstanceOf[DenseVector]))
//    println("=====================kvv:" + fin.getNumPartitions)
//    import spark.implicits._
//    kvv.toDF("label", "features").write.parquet("HIGGS_RSP.parquet")
//  }
}
