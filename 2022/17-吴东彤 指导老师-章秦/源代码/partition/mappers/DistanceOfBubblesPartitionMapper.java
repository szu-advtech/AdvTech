package partition.mappers;

import java.util.LinkedList;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.CoreDistances;
import datastructure.MinimumSpanningTree;
import distance.DistanceCalculator;

public class DistanceOfBubblesPartitionMapper
		implements
		PairFunction<Tuple2<Tuple2<String, ClusterFeatureDataBubbles>, Tuple2<String, ClusterFeatureDataBubbles>>, String, CoreDistances> {

	private static final long serialVersionUID = 1L;
    private DistanceCalculator distanceFunction;
    private double[][] coreDistances;
	
	
	public DistanceOfBubblesPartitionMapper(
			DistanceCalculator distanceFunction, double[][] coreDistancesBubbles) {
		this.coreDistances = coreDistancesBubbles;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, CoreDistances> call(
			Tuple2<Tuple2<String, ClusterFeatureDataBubbles>, Tuple2<String, ClusterFeatureDataBubbles>> t)
			throws Exception {
		double mutual = 0;
		StringBuilder builder = new StringBuilder();
		MinimumSpanningTree cluster = new MinimumSpanningTree();
		String[] key1 = t._1._1().split(" ");
		String[] key2 = t._2._1().split(" ");
		int node1 = Integer.parseInt(key1[0]);
		int node2 = Integer.parseInt(key2[0]);
		int label1 = Integer.parseInt(key1[1]);
		int label2 = Integer.parseInt(key2[1]);
		
		if((label1 != label2) && (node1 == node2)){
			builder.append(label1+" "+label2 + " " +node1);
			double distance = this.distanceFunction.computeDistance(t._1._2.getRep(), t._2._2.getRep());
			double distB = distanceBubbles(distance, t._1._2.getExtent(), t._2._2.getExtent(), t._1._2.getNnDist(), t._2._2.getNnDist());
			int id1 = t._1._2.getId();
			int id2 = t._2._2.getId();
			int pos = checkPositionNode(node1);
			mutual = Math.max(distB, this.coreDistances[pos][id1]);
			mutual = Math.max(mutual, this.coreDistances[pos][id2]);
			cluster.setVertice1(t._1._2.getId());
			cluster.setVertice2(t._2._2.getId());
			cluster.setWeight(mutual);
		}
		else{
			builder.append("_");
		}
		LinkedList<MinimumSpanningTree> mst = new LinkedList<MinimumSpanningTree>();
		mst.add(cluster);
		return new Tuple2(builder.toString(), new CoreDistances(null ,mst));
	}
	
	public int checkPositionNode(int node) {
		int pos = 0;
		for(int i = 0; i < this.coreDistances.length; i++){
			if(this.coreDistances[i][0] == (double) node){
				pos = i;
			}
		}
		return pos;
	}

	public static double distanceBubbles(double distance, double eB, double eC,
			double nnDistB, double nnDistC) {
		double verify = distance - (eB + eC);
		if (verify >= 0) {
			distance = (distance - (eB + eC))
					+ (nnDistB + nnDistC);
		} else {
			distance = Math.max(nnDistB, nnDistC);
		}
		return distance;
	}
	

}
