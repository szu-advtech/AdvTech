package mappers;

import java.util.LinkedList;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.CoreDistances;
import datastructure.MinimumSpanningTree;
import distance.DistanceCalculator;

public class DistanceOfBubblesMapper
		implements
		PairFunction<Tuple2<Tuple2<Integer, ClusterFeatureDataBubbles>, Tuple2<Integer, ClusterFeatureDataBubbles>>, String, CoreDistances> {

	private static final long serialVersionUID = 1L;
	private DistanceCalculator distanceFunction;
	private double[] coreDistances;

	public DistanceOfBubblesMapper(DistanceCalculator dist,
			double[] coreDistances) {
		this.distanceFunction = dist;
		this.coreDistances = coreDistances;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, CoreDistances> call(
			Tuple2<Tuple2<Integer, ClusterFeatureDataBubbles>, Tuple2<Integer, ClusterFeatureDataBubbles>> t)
			throws Exception {
		double mutual = 0;
		StringBuilder builder = new StringBuilder();
		MinimumSpanningTree cluster = new MinimumSpanningTree();
		if (t._1._1().intValue() != t._2._1().intValue()) {
			builder.append(t._1._1().intValue() + " " + t._2._1().intValue());
			double distance = distanceFunction.computeDistance(
					t._1._2.getRep(), t._2._2.getRep());
			double distB = distanceBubbles(distance, t._1._2.getExtent(),
					t._2._2.getExtent(), t._1._2.getNnDist(),
					t._2._2.getNnDist());
			int id1 = t._1._2.getId();
			int id2 = t._2._2.getId();
			mutual = Math.max(distB, this.coreDistances[id1]);
			mutual = Math.max(mutual, this.coreDistances[id2]);
			cluster.setVertice1(t._1._2.getId());
			cluster.setVertice2(t._2._2.getId());
			cluster.setWeight(mutual);
		} else {
			builder.append("_");
		}
		LinkedList<MinimumSpanningTree> mst = new LinkedList<MinimumSpanningTree>();
		mst.add(cluster);
		return new Tuple2(builder.toString(), new CoreDistances(null, mst));
	}

	public static double distanceBubbles(double distance, double eB, double eC,
			double nnDistB, double nnDistC) {
		double verify = distance - (eB + eC);
		if (verify >= 0) {
			distance = (distance - (eB + eC)) + (nnDistB + nnDistC);
		} else {
			distance = Math.max(nnDistB, nnDistC);
		}
		return distance;
	}

}
