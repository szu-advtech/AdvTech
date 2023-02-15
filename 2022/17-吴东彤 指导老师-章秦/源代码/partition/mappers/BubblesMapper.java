package partition.mappers;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataPoints;

public class BubblesMapper implements
		PairFunction<Tuple2<String, ClusterFeatureDataBubbles>, String, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	private Tuple2<String, ClusterFeatureDataBubbles> tuple;

	public Tuple2<String, ClusterFeatureDataBubbles> call(Tuple2<String, ClusterFeatureDataBubbles> t)
			throws Exception {
			String[] node = t._1().split("\\.");
			this.tuple = new Tuple2<String, ClusterFeatureDataBubbles>(node[0], t._2());
		return this.tuple;
	}
}
