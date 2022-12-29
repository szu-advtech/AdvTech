package stream.etl

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object DataTransform {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StreamRSPDataSetTransform").setMaster("yarn")
    etlForHIGGS(conf)
  }

  def etlForHIGGS(sparkConf: SparkConf) = {
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val frame: DataFrame = spark.read.csv("/user/zhaolingxiang/HIGGS.csv")
    val value: RDD[(Double, DenseVector)] = frame.rdd.map(row => {
      val length: Int = row.length
      var label = row.get(0).asInstanceOf[String].toDouble
      if (label < 0) {
        label = 0.0
      }
      val ar = new Array[Double](length - 1)
      for (i <- 1 until length) {
        ar(i - 1) = row.get(i).asInstanceOf[String].toDouble
      }
      (label, new DenseVector(ar))
    })
    import spark.implicits._
    value.repartition(100).toDF("label", "features").write.parquet("HIGGS.parquet")
  }

  def etlForCovertype(sparkConf: SparkConf) = {
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val frame: DataFrame = spark.read.text("/user/zhaolingxiang/covtype.data")
    val value: RDD[(Double, DenseVector)] = frame.rdd.map(row => {
      var features = row.get(0).asInstanceOf[String].split(",").map(_.toDouble)

      val ar = new Array[Double](features.length - 1)
      for (i <- 0 until features.length - 1) {
        ar(i) = features(i)
      }
      (features(features.length - 1), new DenseVector(ar))
    })
    import spark.implicits._
    value.repartition(12).toDF("label", "features").write.parquet("Covertype.parquet")
  }
}
