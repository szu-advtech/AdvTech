package DHBoost.Higges

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.column

import scala.collection.mutable

object TestModel {

    def main(args: Array[String]): Unit = {
        // 日志输出
        val log = new StringBuilder()

        // 创建环境
        val spark: SparkSession = SparkSession
            .builder
            .appName("TestModel_Higges")
            .getOrCreate()

        // 写入配置
        val path = "hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/"
        log.append("job,Test_Higges\n")

        // 加载数据，并进行类型转换
        val dataFrame = spark
            .read.format("csv")
            .load("hdfs://172.31.238.20:8020/user/chandler/recurrence/higges/test/part-00000*.csv") // 加载测试数据
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

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0. 公共变量初始化e>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        var modelPath:String = null
        val n: Int = 10 // 集成使用的模型数
        var i = -1 // 将权重转换成tuple辅助变量

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1. 测试AdaBoostDecisionTree>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 配置模型路径
        modelPath = path + "AdaBoostDecisionTree/"
        // 加载权重向量，以(索引，权重值)存储
        i = -1
        val adaBoostDecisionTreeModelInfo = spark.sparkContext.textFile(modelPath + "weight").map(x => {
            i = i + 1
            (i, x.toDouble)
        }).sortBy(_._2,false).take(n)
        // 加载模型
        val adaBoostDecisionTreeModelArray: Array[DecisionTreeClassificationModel] = new Array[DecisionTreeClassificationModel](n)
        for(i <- 0 until n){
            adaBoostDecisionTreeModelArray(i) = DecisionTreeClassificationModel.read.load(modelPath + adaBoostDecisionTreeModelInfo(i)._1)
        }
        // 集成判断
        val adaBoostDecisionTreeCorrect = dataFrameModify.rdd.map(row => {

            // 使用hashmap存储预测值与权重, key: Label; value: Weight
            val map = new mutable.HashMap[Double, Double]()

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            for (i <- 0 until n) {
                val predict = adaBoostDecisionTreeModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostDecisionTreeModelInfo(i)._2)
                else
                    map.put(predict, adaBoostDecisionTreeModelInfo(i)._2)
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

        // 输出结果
        println(s"adaBoostDecisionTreeCorrect = " + adaBoostDecisionTreeCorrect)
        log.append(s"adaBoostDecisionTreeCorrect," + adaBoostDecisionTreeCorrect + "\n")

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2. 测试AdaBoostLogistic>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 配置模型路径
        modelPath = path + "AdaBoostLogistic/"
        // 加载权重向量，以(索引，权重值)存储，取出前10个值，降序排列
        i = -1
        val adaBoostLogisticModelInfo = spark.sparkContext.textFile(modelPath + "weight").map(x => {
            i = i + 1
            (i, x.toDouble)
        }).sortBy(_._2, false).take(n)
        // 加载模型
        val adaBoostLogisticModelArray: Array[LogisticRegressionModel] = new Array[LogisticRegressionModel](n)
        for (i <- 0 until n) {
            adaBoostLogisticModelArray(i) = LogisticRegressionModel.read.load(modelPath + adaBoostLogisticModelInfo(i)._1)
        }
        // 集成判断
        val adaBoostLogisticCorrect = dataFrameModify.rdd.map(row => {

            // 使用hashmap存储预测值与权重, key: Label; value: Weight
            val map = new mutable.HashMap[Double, Double]()

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            for (i <- 0 until n) {
                val predict = adaBoostDecisionTreeModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostLogisticModelInfo(i)._2)
                else
                    map.put(predict, adaBoostLogisticModelInfo(i)._2)
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

        // 输出结果
        println(s"adaBoostLogisticCorrect = " + adaBoostLogisticCorrect)
        log.append(s"adaBoostLogisticCorrect," + adaBoostLogisticCorrect + "\n")

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>3. 测试AdaBoostSVM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 配置模型路径
        modelPath = path + "AdaBoostSVM/"
        // 加载权重向量，以(索引，权重值)存储，取出前10个值，降序排列
        i = -1
        val adaBoostSVMModelInfo = spark.sparkContext.textFile(modelPath + "weight").map(x => {
            i = i + 1
            (i, x.toDouble)
        }).sortBy(_._2, false).take(n)
        // 加载模型
        val adaBoostLinearSVCModelArray: Array[LinearSVCModel] = new Array[LinearSVCModel](n)
        for (i <- 0 until n) {
            adaBoostLinearSVCModelArray(i) = LinearSVCModel.read.load(modelPath + adaBoostSVMModelInfo(i)._1)
        }
        // 集成判断
        val adaBoostLinearSVCCorrect = dataFrameModify.rdd.map(row => {

            // 使用hashmap存储预测值与权重, key: Label; value: Weight
            val map = new mutable.HashMap[Double, Double]()

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            for (i <- 0 until n) {
                val predict = adaBoostLinearSVCModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostSVMModelInfo(i)._2)
                else
                    map.put(predict, adaBoostSVMModelInfo(i)._2)
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

        // 输出结果
        println(s"adaBoostLinearSVCCorrect = " + adaBoostLogisticCorrect)
        log.append(s"adaBoostLinearSVCCorrect," + adaBoostLogisticCorrect + "\n")

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>4. 测试Spark_GBT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 配置模型路径
        modelPath = path + "GBT/model"
        // 加载模型
        val GBTModel = GBTClassificationModel.read.load(modelPath)

        // 集成判断
        val GBTCorrect = dataFrameModify.rdd.map(row => {

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            val predict = GBTModel.predict(row(0).asInstanceOf[Vector])

            // 判断预测值是否与真实值一致
            if (predict != row(1))
                0.0
            else
                1.0
        }).sum() / datasetSize

        // 输出结果
        println(s"GBTCorrect = " + GBTCorrect)
        log.append(s"GBTCorrect," + GBTCorrect + "\n")

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>5. 测试RandomForest>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 配置模型路径
        modelPath = path + "RandomForest/model"
        // 加载模型
        val RandomForestModel = RandomForestClassificationModel.read.load(modelPath)

        // 集成判断
        val RandomForestCorrect = dataFrameModify.rdd.map(row => {

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            val predict = RandomForestModel.predict(row(0).asInstanceOf[Vector])

            // 判断预测值是否与真实值一致
            if (predict != row(1))
                0.0
            else
                1.0
        }).sum() / datasetSize

        // 输出结果
        println(s"RandomForestCorrect = " + RandomForestCorrect)
        log.append(s"RandomForestCorrect," + RandomForestCorrect + "\n")

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>6. 测试DHBoost_裁剪>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 集成判断
        val DHBoostCorrect = dataFrameModify.rdd.map(row => {

            // 使用hashmap存储预测值与权重, key: Label; value: Weight
            val map = new mutable.HashMap[Double, Double]()

            // 输入模型进行预测，将预测结果、权重进行存储到map中
            for (i <- 0 until n) {
                val predict = adaBoostDecisionTreeModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostDecisionTreeModelInfo(i)._2)
                else
                    map.put(predict, adaBoostDecisionTreeModelInfo(i)._2)
            }
            for (i <- 0 until n) {
                val predict = adaBoostLogisticModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostLogisticModelInfo(i)._2)
                else
                    map.put(predict, adaBoostLogisticModelInfo(i)._2)
            }
            for (i <- 0 until n) {
                val predict = adaBoostLinearSVCModelArray(i).predict(row(0).asInstanceOf[Vector])
                if (map.contains(predict))
                    map.put(predict, map.get(predict).get + adaBoostSVMModelInfo(i)._2)
                else
                    map.put(predict, adaBoostSVMModelInfo(i)._2)
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

        // 输出结果
        println(s"DHBoostCorrect = " + DHBoostCorrect)
        log.append(s"DHBoostCorrect," + DHBoostCorrect + "\n")

        // 写出日志
        spark.sparkContext.parallelize(log.toString().split("\n").toSeq,1).saveAsTextFile(path + "testRes_20")

        //关闭所有东西
        spark.close()
    }
}
