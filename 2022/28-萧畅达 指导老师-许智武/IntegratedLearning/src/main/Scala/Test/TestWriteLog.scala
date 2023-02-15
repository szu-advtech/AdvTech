package Test

import org.apache.spark.sql.SparkSession

object TestWriteLog {

    def main(args: Array[String]): Unit = {

        val spark: SparkSession = SparkSession
            .builder
            .master("local[*]")
            .appName("TestWriteLog")
            .getOrCreate()

        val log:String = "Hello Spark \n" +
            "My name is Chandler\n"

        spark.sparkContext.parallelize(log.split("\n").toSeq,1).saveAsTextFile("D:\\recurrence\\test.txt")
        spark.close()
    }

}
