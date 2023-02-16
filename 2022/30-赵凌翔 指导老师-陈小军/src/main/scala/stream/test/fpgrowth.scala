package stream.test
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partitioner, SparkConf, SparkContext}

import java.io.File
import java.util
import scala.collection.immutable.Set
import scala.util.Random
/**
 * @Author Lingxiang Zhao
 * @Date 2022/12/12 9:55
 * @desc
 */
object fpgrowth {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StreamRSP").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val transaction: RDD[Array[Int]] = sc.textFile("testFile").map(row => row.split(",").map(_.toInt))
    val broadcastList = Array(List(1, 2), List(2, 3))
    val value1: RDD[(String, Int)] = transaction.mapPartitions((arr: Iterator[Array[Int]]) => {
      val temp: Array[List[Int]] = util.Arrays.copyOf(broadcastList, broadcastList.length) //广播数组
      val set: Array[Set[Int]] = arr.map(_.toSet).toArray
      val partitionRes: Array[(String, Int)] = temp.map(items => { //List[Int]
        var count = 0
        for (orginalData <- set) { //List[Int]
          var b = true
          for (item <- items if b) { //Int
            if (!orginalData.contains(item)) {
              b = false
            }
          }
          if (b) {
            count = count + 1
          }
        }
        (items.mkString("{", ",", "}"), count)
      })
      partitionRes.iterator
    })
    println("fenqu:", transaction.getNumPartitions)
    value1.reduceByKey(_+_).map(x => x._1 + ": " + x._2).foreach(println)
  }

}
