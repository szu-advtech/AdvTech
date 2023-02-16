package partition.mappers;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;

public class RetrieveLabelsPartitionsMapperBubbles
		implements
		PairFunction<Tuple2<String, ClusterFeatureDataBubbles>, String, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	private int[][] nodesID;
	private int[][] iBubbles;
	private int[][] clustersID;

	public RetrieveLabelsPartitionsMapperBubbles() {

	}

	public RetrieveLabelsPartitionsMapperBubbles(int[][] idBubbles,
			int[][] clustersID, int[][] nodesID) {
		this.setiBubbles(idBubbles);
		this.setClustersID(clustersID);
		this.setNodesID(nodesID);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, ClusterFeatureDataBubbles> call(
			Tuple2<String, ClusterFeatureDataBubbles> t) throws Exception {
		int currentIdBubble = t._2.getId();
		String[] key = t._1().split("\\.");
		int nodeCurrentBubble = Integer.parseInt(key[0]);
		int label = 0;
		int node = 0;

		for (int i = 0; i < this.clustersID.length; i++) {
			for (int j = 0; j < clustersID[i].length; j++) {
				if ((nodeCurrentBubble == this.nodesID[i][j])
						&& (currentIdBubble == this.iBubbles[i][j])) {
					label = this.clustersID[i][j];
					node = this.nodesID[i][j];
				}
			}
		}
		StringBuilder builder = new StringBuilder();
		builder.append(node + " " + label);
		return new Tuple2(builder.toString(), t._2());
	}

	public int[][] getNodesID() {
		return nodesID;
	}

	public void setNodesID(int[][] nodesID) {
		this.nodesID = nodesID;
	}

	public int[][] getiBubbles() {
		return iBubbles;
	}

	public void setiBubbles(int[][] iBubbles) {
		this.iBubbles = iBubbles;
	}

	public int[][] getClustersID() {
		return clustersID;
	}

	public void setClustersID(int[][] clustersID) {
		this.clustersID = clustersID;
	}
}
