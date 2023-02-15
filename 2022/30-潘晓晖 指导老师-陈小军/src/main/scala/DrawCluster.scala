package org.example

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartFrame, StandardChartTheme}
import org.jfree.data.xy.DefaultXYDataset

import java.awt.{BasicStroke, Color, Font}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object DrawCluster {
  // 读取文件，获取数据
  def getPoints(pointsPath: String, rhoDeltaPath: String, delim: String, dimension: Int): Array[Point] = {
    val pointsMap = mutable.HashMap[Int, Point]()
    val conf = new SparkConf().setMaster("local[2]").setAppName("draw_cluster")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc = session.sparkContext

    val pValues = sc.textFile(pointsPath).collect()
    for (line <- pValues) {
      val s = line.trim.split(delim)
      val pid = s(0).toInt
      val pv = s.slice(1, dimension + 1).map(_.toDouble)
      val point = new Point(pid, pv, false)
      pointsMap.put(pid, point)
    }

    val rhoDelta = sc.textFile(rhoDeltaPath).collect()
    for (rd <- rhoDelta) {
      val arr = rd.split("#")
      val pid = arr(0).toInt
      val rho = arr(1).toDouble
      val delta = arr(2).toDouble
      val nn = arr(3).toInt
      val point = pointsMap(pid)
      point.rho = rho
      point.delta = delta
      point.NNid = nn
    }

    pointsMap.values.toArray
  }

  // 遍历数据点，根据指定的参数找出所有密度峰并分配聚类编号
  def findDensityPeak(points: Array[Point], minRho: Double, minDelta: Double): Int = {
    var clusterID = 0
    for (point <- points) {
      if (point.rho > minRho && point.delta > minDelta) {
        point.isDP = true
        point.cid = clusterID
        point.NNid = point.pid
        clusterID += 1
      }
    }
    clusterID
  }

  // 根据密度从大到小遍历数据点，分配对应聚类编号
  def assignCluster(points: Array[Point]): Unit = {
    val sortedPoints = points.sorted(Ordering.by[Point, Double](-_.rho))
    val assignedPoints = mutable.HashMap[Int, Point]()
    for (point <- sortedPoints) {
      val nnid = point.NNid
      if (assignedPoints.contains(nnid)) {
        val nn = assignedPoints(nnid)
        point.cid = nn.cid
      }
      assignedPoints.put(point.pid, point)
    }
  }

  // 画散点图，展示聚类结果
  def drawCluster(points: Array[Point]): Unit = {
    val clusters = mutable.HashMap[Int, ArrayBuffer[Point]]()
    for (point <- points) {
      if (clusters.contains(point.cid)) {
        clusters(point.cid).append(point)
      } else {
        clusters.put(point.cid, ArrayBuffer[Point](point))
      }
    }
    val xyDataset = new DefaultXYDataset()
    for (cid <- clusters.keys) {
      val pointsInCluster = clusters(cid).toArray
      val size = pointsInCluster.length
      val data = Array.ofDim[Double](2, size)
      for (i <- 0 until size) {
        data(0)(i) = pointsInCluster(i).pv(0)
        data(1)(i) = pointsInCluster(i).pv(1)
      }
      val label = "c" + cid
      xyDataset.addSeries(label, data)
    }

    // 创建主题样式
    val mChartTheme = new StandardChartTheme("CN")
    // 设置标题字体
    mChartTheme.setExtraLargeFont(new Font("黑体", Font.BOLD, 20))
    // 设置轴向字体
    mChartTheme.setLargeFont(new Font("宋体", Font.CENTER_BASELINE, 15))
    // 设置图例字体
    mChartTheme.setRegularFont(new Font("宋体", Font.CENTER_BASELINE, 15))
    // 应用主题样式
    ChartFactory.setChartTheme(mChartTheme)

    // 创建散点图
    val chart = ChartFactory.createScatterPlot("聚类结果", "x", "y",
      xyDataset, PlotOrientation.VERTICAL, false, false, false)
    val frame = new ChartFrame("散点图", chart, true)
    chart.setBackgroundPaint(Color.white)
    chart.setBorderPaint(Color.GREEN)
    chart.setBorderStroke(new BasicStroke(1.5f))
    chart.getPlot.setBackgroundPaint(new Color(255, 253, 246))

    frame.pack()
    frame.setVisible(true)
  }

  private def printUsage(): Unit = {
    println("<points input path> <rho_delta input path> <point dimensions> <min_rho> <min_delta>")
    println("\t-d delim\n")
  }

  def main(args: Array[String]): Unit = {
    // check and get parameters
    if (args.length < 5) {
      printUsage()
      return
    }
    var delim: String = " "
    var pointsPath: String = "D:/panxh/Documents/lsh_test/aggregation.txt"
    var rhoAndDeltaPath: String = "D:/panxh/Documents/lsh_test/corrected_result.txt"
    var dimension: Int = 2
    var minRho: Double = 0
    var minDelta: Double = 0
    val otherArgs = new ArrayBuffer[String]()

    var index = 0
    while (index < args.length) {
      if ("-d".equals(args(index))) {
        index += 1
        delim = args(index)
      } else {
        otherArgs += args(index)
      }
      index += 1
    }

    if (otherArgs.length < 5) {
      println("ERROR: Wrong number of parameters: " + otherArgs.length + ".")
      printUsage()
      return
    }

    try {
      pointsPath = otherArgs(0)
      rhoAndDeltaPath = otherArgs(1)
      dimension = otherArgs(2).toInt
      minRho = otherArgs(3).toDouble
      minDelta = otherArgs(4).toDouble
    } catch {
      case _: NumberFormatException =>
        println("ERROR: Wrong type of parameters.")
        printUsage()
        return
    }

    // 获取数据并分配聚类中心
    val points = getPoints(pointsPath, rhoAndDeltaPath, delim, dimension)
    println("成功读取数据")
    val clusterNum = findDensityPeak(points, minRho, minDelta)
    println("成功找出密度峰点，一共有 " + clusterNum + " 个")
    assignCluster(points)
    println("成功分配聚类编号")

    // 画图
    println("开始画图")
    drawCluster(points)
  }

}
