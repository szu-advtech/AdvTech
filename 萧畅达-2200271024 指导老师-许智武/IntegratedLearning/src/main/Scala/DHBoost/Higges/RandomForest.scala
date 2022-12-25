package DHBoost.Higges

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.column

/**
 * spark_随机森林分类器
 * */
object RandomForest {

    def main(args: Array[String]): Unit = {
        // 开始计时
        val startTime = System.currentTimeMillis()

        // 创建环境
        val spark: SparkSession = SparkSession
            .builder
            .appName("RandomForest_Higges")
            .getOrCreate()

        // 日志输出
        val log = new StringBuilder()

        // 写入配置
        val path = "hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/RandomForest/"
        log.append("job,GBT_Higges\n")

        // 加载数据，并进行类型转换
        val dataFrame = spark
            .read.format("csv")
            .load("hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/train/part-00000*.csv") // 加载所有数据
            .toDF("label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
                "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28")
            .withColumn("label", column("label").cast("Double"))
            .withColumn("f1", column("f1").cast("Double"))
            .withColumn("f2", column("f2").cast("Double"))
            .withColumn("f3", column("f3").cast("Double"))
            .withColumn("f4", column("f4").cast("Double"))
            .withColumn("f5", column("f5").cast("Double"))
            .withColumn("f6", column("f6").cast("Double"))
            .withColumn("f7", column("f7").cast("Double"))
            .withColumn("f8", column("f8").cast("Double"))
            .withColumn("f9", column("f9").cast("Double"))
            .withColumn("f10", column("f10").cast("Double"))
            .withColumn("f11", column("f11").cast("Double"))
            .withColumn("f12", column("f12").cast("Double"))
            .withColumn("f13", column("f13").cast("Double"))
            .withColumn("f14", column("f14").cast("Double"))
            .withColumn("f15", column("f15").cast("Double"))
            .withColumn("f16", column("f16").cast("Double"))
            .withColumn("f17", column("f17").cast("Double"))
            .withColumn("f18", column("f18").cast("Double"))
            .withColumn("f19", column("f19").cast("Double"))
            .withColumn("f20", column("f20").cast("Double"))
            .withColumn("f21", column("f21").cast("Double"))
            .withColumn("f22", column("f22").cast("Double"))
            .withColumn("f23", column("f23").cast("Double"))
            .withColumn("f24", column("f24").cast("Double"))
            .withColumn("f25", column("f25").cast("Double"))
            .withColumn("f26", column("f26").cast("Double"))
            .withColumn("f27", column("f27").cast("Double"))
            .withColumn("f28", column("f28").cast("Double"))

        // 封装dataFrame成(feature,label)形式
        var dataFrameModify = new VectorAssembler()
            .setInputCols(Array("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
                "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28"))
            .setOutputCol("feature")
            .transform(dataFrame)
            .select("feature", "label")

        // 记录数据集大小和标签数
        val datasetSize: Double = dataFrameModify.count().toDouble
        val labelSize: Double = dataFrameModify.select("label").distinct().count().toDouble
        log.append(s"datasetSize," + datasetSize + "\n")
            .append(s"labelSize," + labelSize + "\n")

        // 初始化一些错误率等参数：迭代次数、错误率1、错误率2、权重之和； alpha为当前模型的权重 beta为更新权重用的参数，alpha和beta有联系
        var beta = 0.0

        val model = new RandomForestClassifier()
            .setFeaturesCol("feature")
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .fit(dataFrameModify)

        dataFrameModify = model.transform(dataFrameModify)

        // 计算本次生成的分类器的分类准确率, 得出错误率
        val correct = dataFrameModify.select("label", "prediction").rdd.map(row => {
            if (row(0) != row(1))
                0.0
            else
                1.0
        }).sum() / datasetSize
        val error = 1.0 - correct

        // 停止计时
        val endTime = System.currentTimeMillis()
        val costTime = (endTime - startTime) / 1000
        log.append("costTime," + costTime + "s\n")
        log.append("correct," + correct + "\n")

        // 公式中的β值
        beta = error / (1 - error)
        log.append(s"beta," + beta + "\n")
            .append(s"error," + error + "\n")

        model.write.overwrite().save(path + "model")
        spark.sparkContext.parallelize(log.toString().split("\n").toSeq, 1).saveAsTextFile(path + "log")

        // 停止环境
        spark.stop()
    }
}
