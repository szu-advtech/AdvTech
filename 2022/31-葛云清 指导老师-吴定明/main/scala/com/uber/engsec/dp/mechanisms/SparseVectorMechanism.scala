package com.uber.engsec.dp.mechanisms


class SparseVectorMechanism(epsilon: Double, queries: List[Relation], threshold: Double,
  config: RewriterConfig) extends ChorusMechanism[Option[Int]] {

  def maxSensitivity(query: Relation, database: Database): Double = {
    val facts = new GlobalSensitivityAnalysis().run(query, database).colFacts
    facts.map(_.sensitivity.get).max
  }

  def run(): (Option[Int], EpsilonDPCost) = {
    val cost = EpsilonDPCost(epsilon)

    // Require queries to have a maximum sensitivity of 1
    val sensitivities = queries.map { (q: Relation) => maxSensitivity(q, config.database) }
    if (sensitivities.max > 1)
      return (None, cost)

    // Generate noisy threshold
    val T = threshold + BasicMechanisms.laplaceSample(2/epsilon)

    // Loop over the queries, to find one exceeding the threshold
    for (i <- 0 to queries.length) {
      DB.execute(queries(i), config.database) match {
        case List(DB.Row(List(r))) =>
          if (r.toDouble + BasicMechanisms.laplaceSample(4/epsilon) >= T)
            return (Some(i), cost)
      }
    }

    (None, cost)
  }
}

class SparseVectorMechanismValue(epsilon: Double, queries: List[Relation],
  threshold: Double, config: RewriterConfig) extends ChorusMechanism[List[DB.Row]] {

  def run() = {
    val (Some(idx), c1) = new SparseVectorMechanism(epsilon/2, queries, threshold, config).run()
    val (result, c2) = new LaplaceMechClipping(epsilon/2, 0, 10, queries(idx), config).run()
    
    (result, c1 + c2)
  }
}
