package Remote

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object WC {
  //指定服务器部署ops-master节点发起spark-submit
  private val master = "spark://Spark01:7077"
  private val remote_file = "file:///D:\\fandaima\\SPARK\\input\\*.txt"
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("WordCount")
      .setMaster(master)
      .setJars(List("D:\\fandaima\\SPARK\\target\\spark-parent_2.11-2.4.3-tests.jar"))

    val sc = new SparkContext(conf)
    val textFile1 = sc.textFile(remote_file)
    val wordCount = textFile1.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey((a, b) => a + b)
    wordCount.foreach(println)
  }
}
