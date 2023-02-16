package com.uber.engsec.dp.mechanisms


class ReportNoisyMax(epsilon: Double, queries: List[Relation], config: RewriterConfig)
    extends ChorusMechanism[Int] {

  def run() = {
    val results = queries.map { (q: Relation) =>
      new LaplaceMechClipping(epsilon, 0, 1, q, config).run()._1 }
    val unwrappedResults : List[Double] =
      results.map { case List(DB.Row(List(i))) => i.toDouble }

    (BasicMechanisms.argmax(unwrappedResults), EpsilonDPCost(epsilon))
  }
}

