package stream.algo

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.util.Random

object LogisticRegressionSpark {

  var maxIter: Int = 500
  var regParam: Double = 0.1

  def SparkLinearRegressionClassification(train:DataFrame, test:DataFrame): (Double, Double) = {

    //建立模型
    val logisticRegression: LogisticRegression = new LogisticRegression()
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setLabelCol("label")
      .setFeaturesCol("features")

    //建立工作流
    val pipeline = new Pipeline().setStages(Array(logisticRegression))

    //开始计时
    val startTime = System.nanoTime

    //训练
    val logisticRegressionModel = pipeline.fit(train)

    //结束计时&计算训练耗时
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    //测试数据预测
    val result = logisticRegressionModel.transform(test)

    val rightCount= result.select("label", "prediction").rdd.filter(x => x.getDouble(0) == x.getDouble(1)).count()

    (duration, rightCount.toDouble / result.count())
    }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegression-Classification-StreamRSP").setMaster("yarn")
    val sc = new SparkContext(conf)
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val files = scala.collection.mutable.ListBuffer.empty[String]
    val fs = FileSystem.get(sc.hadoopConfiguration)
    //fs.listStatus(new Path("/user/zhaolingxiang/Covertype_RSP.parquet")).foreach(x => files += x.getPath.toString)
    fs.listStatus(new Path("/user/zhaolingxiang/HIGGS_RSP.parquet")).foreach(x => files += x.getPath.toString)
    val paths: Array[String] = files.toArray
    val RSPtrain: DataFrame = spark.read.parquet(paths(new Random().nextInt(paths.length - 1) + 1)).cache()
    //val Array(train, test): Array[Dataset[Row]] = spark.read.parquet("/user/zhaolingxiang/Covertype.parquet").cache().randomSplit(Array(0.9, 0.1))
    val Array(train, test): Array[Dataset[Row]] = spark.read.parquet("/user/zhaolingxiang/HIGGS.parquet").cache().randomSplit(Array(0.9, 0.1))
    val result = SparkLinearRegressionClassification(train, test)
    val rspResult = SparkLinearRegressionClassification(RSPtrain, test)
    println("训练耗时:" + result._1 + "s")
    println("预测准确度:" + result._2)
    println("RSP训练耗时:" + rspResult._1 + "s")
    println("RSP预测准确度:" + rspResult._2)
    spark.stop()
  }
}
