package DHBoost.Susy
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataPartition {

    def main(args: Array[String]): Unit = {
        // 开始计时
        val startTime = System.currentTimeMillis()

        // 初始化参数
        val L: Int = 3; //基学习器数量与分区数量一致
        // 创建环境
        val spark: SparkSession = SparkSession
            .builder
            .appName("DataPartition_Susy")
            .getOrCreate()

        // 对数据进行切分处理
        println("DataPartition\n")

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

        splitData(1)
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
        println("costTime = " + costTime)

        // 停止
        spark.stop()
    }
}
