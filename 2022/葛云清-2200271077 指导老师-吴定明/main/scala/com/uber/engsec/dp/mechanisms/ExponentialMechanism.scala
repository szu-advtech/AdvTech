package com.uber.engsec.dp.mechanisms


class ExponentialMechanism(epsilon: Double, scoring: Relation, config: RewriterConfig)
    extends ChorusMechanism[Int] {

  def run() = {
    val sensitivity = new GlobalSensitivityAnalysis().run(scoring, config.database)
      .colFacts.map(_.sensitivity.get).max

    val scores = DB.execute(scoring, config.database)
    val totalScore = scores.map { case DB.Row(List(_, v)) => v.toDouble }.sum

    val probabilities = scores.map { case DB.Row(List(k, v)) =>
      (k, epsilon * (v.toDouble / totalScore) / (2 * sensitivity)) }

    (BasicMechanisms.chooseWithProbability(probabilities),
      EpsilonDPCost(epsilon))
  }
}
