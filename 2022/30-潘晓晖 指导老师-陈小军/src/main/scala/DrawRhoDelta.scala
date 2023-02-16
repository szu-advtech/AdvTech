package org.example

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartFrame, StandardChartTheme}
import org.jfree.data.xy.DefaultXYDataset

import java.awt.{BasicStroke, Color, Font}

object DrawRhoDelta {
  def createXYDataset(filePath: String): DefaultXYDataset = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("draw_rho_delta")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc = session.sparkContext
    val lines = sc.textFile(filePath).collect()
    val xyDataset = new DefaultXYDataset()
    val data = Array.ofDim[Double](2, lines.length)
    for (i <- lines.indices) {
      if (lines(i).nonEmpty) {
        val arr = lines(i).split("#")
        val rho = arr(1).toDouble
        val delta = arr(2).toDouble
        data(0)(i) = rho
        if (delta == Double.MaxValue) {
          data(1)(i) = 0
        } else {
          data(1)(i) = delta
        }
      }
    }
    val index = data(1).indexOf(0)
    data(1)(index) = data(1).max
    xyDataset.addSeries("rho&delta", data)
    xyDataset
  }

  def main(args: Array[String]): Unit = {
    // 读取文件，获取数据
    val inputPath = args(0)
    val xyDataset = createXYDataset(inputPath)
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
    println("creating scatterPlot...")
    val chart = ChartFactory.createScatterPlot("决策图", "Rho", "Delta",
      xyDataset, PlotOrientation.VERTICAL, false, false, false)
    val frame = new ChartFrame("散点图", chart, true)
    chart.setBackgroundPaint(Color.white)
    chart.setBorderPaint(Color.GREEN)
    chart.setBorderStroke(new BasicStroke(1.5f))
    chart.getPlot.setBackgroundPaint(new Color(255, 253, 246))

    // 画图
    println("drawing the figure...")
    frame.pack()
    frame.setVisible(true)
  }
}
