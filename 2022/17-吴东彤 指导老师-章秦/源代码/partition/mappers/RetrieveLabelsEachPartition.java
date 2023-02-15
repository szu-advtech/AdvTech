package partition.mappers;

import java.util.LinkedList;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataPoints;
import datastructure.DataBubbles;

public class RetrieveLabelsEachPartition implements
		PairFunction<Tuple2<String, ClusterFeatureDataBubbles>, String, DataPoints> {

	private static final long serialVersionUID = 1L;
	private DataBubbles[] dataBubbles;

	public RetrieveLabelsEachPartition(DataBubbles[] dataBubbles) {
		this.setDataBubbles(dataBubbles);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, DataPoints> call(Tuple2<String, ClusterFeatureDataBubbles> t)
			throws Exception {
		int node = 0;
		int size = 0;

		StringBuilder key = new StringBuilder();
			double[] currentPoint = t._2.getPoints().getPoint();
			int currentIdPoint = t._2.getPoints().getIdPoint();
			int currentIdBubble = t._2.getPoints().getIdBubble();
			String[] currentNodePointS = t._1().split("\\.");
			int currentNodePoint = Integer.parseInt(currentNodePointS[0]);

			for (int i = 0; i < this.dataBubbles.length; i++) {
				for (int j = 0; j < this.dataBubbles[i].getKeys().length; j++) {
					String[] nodeBubble = dataBubbles[i].getKeys()[j].split("\\.");
					if ((currentNodePoint == Integer.parseInt(nodeBubble[0])) && (currentIdBubble == dataBubbles[i].getIdB()[j])) {
						node = this.dataBubbles[i].getNodesID()[j];
						size = this.dataBubbles[i].getTamClusters()[j];
					}
				}
			}
			key.append(node + "." + "0" + "." + size);
			Tuple2<String, DataPoints> tuple = new Tuple2(node, new DataPoints(currentIdPoint,
					currentIdBubble, key.toString(), currentPoint));
		return tuple;
	}

	public DataBubbles[] getDataBubbles() {
		return dataBubbles;
	}

	public void setDataBubbles(DataBubbles[] dataBubbles) {
		this.dataBubbles = dataBubbles;
	}
}
