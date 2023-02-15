package main;

import java.util.List;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import scala.Tuple3;

public class FilterTiedEdges implements Function<Tuple2<Integer, Tuple3<Integer, Integer, Double>>, Boolean> {

	private static final long serialVersionUID = 1L;
	private List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> higher;

	public FilterTiedEdges(List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> higher) {
		this.higher = higher;
	}

	public Boolean call(Tuple2<Integer, Tuple3<Integer, Integer, Double>> v1) throws Exception {
		for(int i = 0; i < this.higher.size(); i++){
			if(v1._2()._3().doubleValue() == this.higher.get(i)._2()._3().doubleValue()){
				return true;
			} 
		}
		return false;
	}
}
