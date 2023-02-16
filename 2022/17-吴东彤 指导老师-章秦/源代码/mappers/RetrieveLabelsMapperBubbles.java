package mappers;


import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataBubbles;

public class RetrieveLabelsMapperBubbles implements PairFunction<Tuple2<Integer, ClusterFeatureDataBubbles>, Integer, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	private DataBubbles dataBubbles;
	
	public RetrieveLabelsMapperBubbles(DataBubbles dataBubbles) {
		this.setDataBubbles(dataBubbles);
	}
    
	public void setDataBubbles(DataBubbles dataBubbles){
		this.dataBubbles = dataBubbles;
	}
	
	public DataBubbles getDataBubbles(){
		return this.dataBubbles;
	}
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<Integer,ClusterFeatureDataBubbles> call(Tuple2<Integer,ClusterFeatureDataBubbles> t) throws Exception {
		int currentIdBubble = t._2.getId();
		int label = 0;
		
		for(int i = 0; i < this.dataBubbles.getIdCluster().length; i++){
			if(currentIdBubble == this.dataBubbles.getIdB()[i]){
				label = this.dataBubbles.getIdCluster()[i];
			}
		}
		return new Tuple2(label, t._2());
	}
}
