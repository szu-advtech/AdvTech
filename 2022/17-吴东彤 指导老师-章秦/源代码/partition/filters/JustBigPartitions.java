package partition.filters;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.DataPoints;

public class JustBigPartitions implements
		Function<Tuple2<String, DataPoints>, Boolean> {

	private static final long serialVersionUID = 1L;
    private int processingU;
    
    public JustBigPartitions(int processingU){
    	this.processingU = processingU;
    }
	
	public Boolean call(Tuple2<String, DataPoints> v1) throws Exception {
		String[] key = v1._2().getKey().split("\\.");
		int size = Integer.parseInt(key[2]);
		return size > this.processingU;
	}
}
