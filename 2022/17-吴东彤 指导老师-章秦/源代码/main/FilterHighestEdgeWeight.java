package main;

import java.util.List;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import scala.Tuple3;

public class FilterHighestEdgeWeight implements Function<Tuple2<Integer, Tuple3<Integer, Integer, Double>>, Boolean> {

	private static final long serialVersionUID = 1L;
	List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> highestEdgeWeight;

	public FilterHighestEdgeWeight(List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> highestEdgeWeight) {
		this.highestEdgeWeight = highestEdgeWeight;
	}

	public Boolean call(Tuple2<Integer, Tuple3<Integer, Integer, Double>> v1) throws Exception {
		boolean situation = true;
		for(int i = 0; i < this.highestEdgeWeight.size(); i++){
			if((v1._2()._1().intValue() == this.highestEdgeWeight.get(i)._2()._1().intValue())
			&& (v1._2()._2().intValue() == this.highestEdgeWeight.get(i)._2()._2().intValue())){
				situation = false;
			}
		}
		return situation;
	}
}
