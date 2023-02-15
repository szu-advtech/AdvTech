package Test

import org.apache.spark.sql.SparkSession

object TestHadoop {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .getOrCreate()

        val sc = spark.sparkContext

        val dataFrame = spark
            .read
            .format("csv")
            .load("hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/SUSY.csv")

        dataFrame.show(20)

        spark.close()
    }

}
