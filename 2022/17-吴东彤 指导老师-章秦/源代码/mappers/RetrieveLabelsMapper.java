package mappers;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataBubbles;
import datastructure.DataPoints;

public class RetrieveLabelsMapper implements PairFunction<Tuple2<Integer, ClusterFeatureDataBubbles>, String, DataPoints> {

	private static final long serialVersionUID = 1L;
	private DataBubbles dataBubbles;
	
	public RetrieveLabelsMapper(DataBubbles dataBubbles) {
		this.setDataBubbles(dataBubbles);
	}
    
	public void setDataBubbles(DataBubbles dataBubbles){
		this.dataBubbles = dataBubbles;
	}
	
	public DataBubbles getDataBubbles(){
		return this.dataBubbles;
	}
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, DataPoints> call(Tuple2<Integer,ClusterFeatureDataBubbles> t) throws Exception {
		double[] currentPoint = t._2.getPoints().getPoint();
		int currentIdPoint = t._2.getPoints().getIdPoint();
		int currentIdBubble = t._2.getPoints().getIdBubble();
		int node = 0;
		int size = 0;
		
		for(int i = 0; i < this.dataBubbles.getIdCluster().length; i++){
			if(currentIdBubble == this.dataBubbles.getIdB()[i]){
				node = this.dataBubbles.getNodesID()[i];
				size = this.dataBubbles.getTamClusters()[i];
			}
		}
		StringBuilder key = new StringBuilder();
		key.append(node+ "." + "0" + "." +size);
		return new Tuple2(node, new DataPoints(currentIdPoint, currentIdBubble, key.toString(), currentPoint));
	}
}
