package DHBoost.Higges

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, column, lit, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable;

/**
 * 逻辑回归分类 + AdaBoost
 * */
object AdaBoostLogistic {

    def main(args: Array[String]): Unit = {
        // 开始计时
        val startTime = System.currentTimeMillis()

        // 日志输出
        val log = new StringBuilder()

        // 创建环境
        val spark: SparkSession = SparkSession
            .builder
            .appName("AdaBoostLogistic_Higges")
            .getOrCreate()

        // 写入配置
        val path = "hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/AdaBoostLogistic/"
        log.append("job,AdaBoostLogistic_Higges\n")

        // 加载数据，并进行类型转换
        val dataFrame = spark
            .read.format("csv")
            .load("hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/train/repartationRdd/part-00001*.csv")// 加载第1个分区
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
        val dataFrameModify = new VectorAssembler()
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

        // 初始化各子模型及其权重
        val maxIter: Int = 20
        val modelWeights: Array[Double] = new Array[Double](maxIter)
        val modelCorrects: Array[Double] = new Array[Double](maxIter)
        val modelErrors: Array[Double] = new Array[Double](maxIter)
        val modelCorrectsSum: Array[Double] = new Array[Double](maxIter)
        val modelArray: Array[LogisticRegressionModel] = new Array[LogisticRegressionModel](maxIter)

        // 初始化一些错误率等参数：迭代次数、错误率1、错误率2、权重之和； alpha为当前模型的权重 beta为更新权重用的参数，alpha和beta有联系
        var iter = 0
        var alpha = 0.0
        var beta = 0.0
        var weightSum = datasetSize

        // 对每个样本计算预测的label和真实label，并根据该样本的现有权重obsWeight进行更新，
        // 如果分类正确，其权重不变，否则增大其权重
        // dataWeight是一个函数
        val dataWeight: (Double, Double, Double) => Double = (obsWeight: Double, labelIndex: Double, prediction: Double) => {
            if (labelIndex == prediction) {
                obsWeight
            }
            else {
                obsWeight / beta
            }
        }

        //使用udf对单个函数进行处理，使之可以对整列数据进行处理
        val sqlfunc = udf(dataWeight)

        // 初始化还没有prediction，初始化所有样本为等权重，归一化
        var temp: DataFrame = dataFrameModify.withColumn("obsWeights", lit(1.0))

        // adaboost迭代过程
        while (iter < maxIter) {

            // 在当前样本权重情况下调用模型进行训练的到当前迭代下的子模型
            val model = new LogisticRegression()
                .setFeaturesCol("feature")
                .setLabelCol("label")
                .setWeightCol("obsWeights")
                .setPredictionCol("prediction")
                .fit(temp)
            temp = model.transform(temp).cache()

            // 计算该模型的错误率 (伊普斯龙) 需要归一化
            val error = temp.select("label", "prediction", "obsWeights").rdd.map(row => {
                if (row(0) != row(1))
                    row.getDouble(2)
                else
                    0.0
            }
            ).sum() / weightSum
            modelErrors(iter) = error

            // 计算本次生成的分类器的分类准确率
            val correct = temp.select("label", "prediction").rdd.map(row => {
                if (row(0) != row(1))
                    0.0
                else
                    1.0
            }).sum() / datasetSize
            modelCorrects(iter) = correct

            // 公式中的β值
            beta = error / (1 - error)
            log.append(s"iter," + iter + "\n")
                .append(s"beta," + beta + "\n")
                .append(s"error," + error + "\n")

            // 权重
            alpha = Math.log(1 / beta)

            if (alpha < 0) {
                temp = temp.withColumn("obsWeights", sqlfunc(col("obsWeights"), col("label"), col("prediction")))
                weightSum = temp.select("obsWeights").rdd.map(row => row.getDouble(0)).sum()
                temp = temp.drop("prediction", "rawPrediction", "probability")
                log.append("info,误差值大于0.5舍弃本次结果\n")
            } else {
                // 保存当前子模型和权重
                modelWeights(iter) = alpha
                modelArray(iter) = model
                model.write.overwrite().save(path + iter)

                // 计算本次集成后的准确率
                // 获取数据集，进行评估
                modelCorrectsSum(iter) = dataFrameModify.rdd.map(row => {

                    // 使用hashmap存储预测值与权重, key: Label; value: Weight
                    val map = new mutable.HashMap[Double, Double]()

                    // 输入模型进行预测，将预测结果、权重进行存储到map中
                    for (i <- 0 to iter) {
                        val predict = modelArray(i).predict(row(0).asInstanceOf[Vector])
                        if (map.contains(predict))
                            map.put(predict, map.get(predict).get + modelWeights(i))
                        else
                            map.put(predict, modelWeights(i))
                    }

                    // 以权重最大的预测结果作为集成结果
                    var maxWeight: Double = 0
                    var currentPredict: Double = -1
                    map.foreach(iterator => {
                        if (iterator._2 > maxWeight) {
                            maxWeight = iterator._2
                            currentPredict = iterator._1
                        }
                    })

                    // 判断最后的预测值是否与真实值一致
                    if (currentPredict != row(1))
                        0.0
                    else
                        1.0
                }).sum() / datasetSize

                // 更新权重
                temp = temp.withColumn("obsWeights", sqlfunc(col("obsWeights"), col("label"), col("prediction")))
                // 计算新的权重和
                weightSum = temp.select("obsWeights").rdd.map(row => row.getDouble(0)).sum()
                // 去掉本轮的预测结果
                temp = temp.drop("prediction", "rawPrediction", "probability")
                iter += 1
            }
        }
        // 停止计时
        val endTime = System.currentTimeMillis()
        val costTime = (endTime - startTime) / 1000
        println("costTime: " + costTime)
        log.append("costTime," + costTime + "s\n")

        // 输出到控制台
        println("Model")
        modelArray.foreach(println).toString
        println(">>>>>>>>>>>>>>>>>>>>>>>>>")
        println("ModelWeight")
        modelWeights.foreach(println).toString
        println(">>>>>>>>>>>>>>>>>>>>>>>>>")
        println("EachModelCorrect")
        modelCorrects.foreach(println)
        println(">>>>>>>>>>>>>>>>>>>>>>>>>")
        println("SumModelCorrect")
        modelCorrectsSum.foreach(println)
        println(">>>>>>>>>>>>>>>>>>>>>>>>>")
        println("EachModelError")
        modelErrors.foreach(println)

        // 保存到文件
        spark.sparkContext.parallelize(log.toString().split("\n").toSeq, 1).saveAsTextFile(path + "log")
        spark.sparkContext.parallelize(modelWeights.toSeq, 1).saveAsTextFile(path + "weight")
        spark.sparkContext.parallelize(modelCorrects.toSeq, 1).saveAsTextFile(path + "correct")
        spark.sparkContext.parallelize(modelCorrectsSum.toSeq, 1).saveAsTextFile(path + "sumCorrect")
        spark.sparkContext.parallelize(modelErrors.toSeq, 1).saveAsTextFile(path + "error")

        // 停止环境
        spark.stop()
    }
}
