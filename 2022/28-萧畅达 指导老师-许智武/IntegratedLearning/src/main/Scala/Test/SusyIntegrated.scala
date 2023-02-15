package Test

import org.apache.spark.sql.{DataFrame, SparkSession}

import java.text.SimpleDateFormat
import java.util.Date

object SusyIntegrated {

    def main(args: Array[String]): Unit = {
        // 开始计时
        val startTime = System.currentTimeMillis()

        // 写入配置
        val path = "hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/"
        val log = new StringBuilder()

        // 初始化参数
        val L: Int = 3; //基学习器数量与分区数量一致
        val T: Int = 100; //迭代次数
        // 创建环境
        val spark: SparkSession = SparkSession
            .builder
            .appName("SusyIntegrated")
            .getOrCreate()

        // 对数据进行切分处理
        log.append("DataPartition\n")

        // 加载数据
        val totalData: DataFrame = spark
            .read
            .format("csv")
            .load("hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/SUSY.csv")

        // 切分测试集训练集
        val splitData = totalData.randomSplit(Array(0.3, 0.7))
        splitData(0)
            .repartition(1)
            .write
            .mode("overwrite")
            .format("csv")
            .save("hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/test") // 测试数据集

        splitData(0)
            .repartition(1)
            .write
            .mode("overwrite")
            .format("csv")
            .save("hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/train") // 训练数据集

        // 数据分区
        splitData(1)
            .repartition(L)
            .write
            .mode("overwrite")
            .format("csv")
            .save("hdfs://172.31.238.20:8020/user/chandler/recurrence/susy/train/repartationRdd") // 手动分区，并进行重命名

        // 停止计时
        val endTime = System.currentTimeMillis()
        val costTime = (endTime - startTime) / 1000
        log.append("costTime = " + costTime + "s\n")

        //写入日志
        spark.sparkContext.parallelize(log.toString(), 1)
            .saveAsTextFile(path + new SimpleDateFormat("yyyy_MM_dd_HH_MM_SS").format(new Date()) + "_log.txt")

        // 停止
        spark.stop()
    }
}
