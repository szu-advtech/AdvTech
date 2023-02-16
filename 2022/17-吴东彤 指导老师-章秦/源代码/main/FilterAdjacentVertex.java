package main;

import java.util.List;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import scala.Tuple3;

public class FilterAdjacentVertex implements PairFunction<Tuple2<Integer, int[]>, Integer, int[]> {

	private static final long serialVersionUID = 1L;
	private List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> highestEdgeWeight;

	public FilterAdjacentVertex(List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> highestEdgeWeight) {
		this.highestEdgeWeight = highestEdgeWeight;
	}

	public Tuple2<Integer, int[]> call(Tuple2<Integer, int[]> t) throws Exception {
		int count = 0;
		for (int higher = 0; higher < this.highestEdgeWeight.size(); higher++) {
			count = 0;
			if (this.highestEdgeWeight.get(higher)._2()._1().intValue() == this.highestEdgeWeight.get(higher)._2()._2()
					.intValue()) {
				continue;
			}
			if (t._1().intValue() == this.highestEdgeWeight.get(higher)._2()._1().intValue()) {
				for (int i = 0; i < t._2().length; i++) {
					if (t._2()[i] == this.highestEdgeWeight.get(higher)._2()._2().intValue()) {
						t._2()[i] = -1;
						count++;
					}
				}
			}
		}
		int[] adj = new int[t._2().length - count];
		int index = 0;
		for (int i = 0; i < t._2().length; i++) {
			if (t._2()[i] != -1) {
				adj[index] = t._2()[i];
				index++;
			}
		}
		return new Tuple2<Integer, int[]>(t._1(), adj);
	}

}
