package stream.evaluation

import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
 * @Author Lingxiang Zhao
 * @Date 2022/12/6 19:13
 * @desc
 */
object DataDistribution {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("CovV1Distribution").setMaster("yarn")
    getRSPDistribution(conf)
  }
  def getTotalDistribution(sparkConf: SparkConf) = {
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val frame: DataFrame = spark.read.parquet("/user/zhaolingxiang/Covertype.parquet")
    val map: RDD[(Double, Int)] = frame.rdd.map(row => {
      (row.get(1).asInstanceOf[DenseVector].toArray(0), 1)
    })
    val res: RDD[(Int, Int)] = map.groupByKey().map(group => {
      (group._1.toInt, group._2.size)
    })
    import spark.implicits._
    res.toDF("data", "count").repartition(1).write.csv("CovertypeTotalDistribution.csv")
  }

  def getRSPDistribution(sparkConf: SparkConf) = {
    val sc = new SparkContext(sparkConf)
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val files = scala.collection.mutable.ListBuffer.empty[String]
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val status: Array[FileStatus] = fs.listStatus(new Path("/user/zhaolingxiang/Covertype_RSP.parquet"))
    status.foreach(x => files += x.getPath.toString)
    val paths: Array[String] = files.toArray
    val ints: List[Int] = List.range(1, paths.length - 1)
    val ints1: List[Int] = Random.shuffle(ints)
    for(i <- 0 to 3){
      val frame: DataFrame = spark.read.parquet(paths(ints1(i)))
      val map: RDD[(Double, Int)] = frame.rdd.map(row => {
        (row.get(1).asInstanceOf[DenseVector].toArray(0), 1)
      })
      val res: RDD[(Int, Int)] = map.groupByKey().map(group => {
        (group._1.toInt, group._2.size)
      })
      import spark.implicits._
      res.toDF("data", "count").repartition(1).write.csv("CovertypeRSPDistribution_Block_" + ints1(i) +".csv")
    }
  }
}
