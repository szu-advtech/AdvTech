package main;

import java.util.List;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import scala.Tuple3;

public class LabelClassification implements
		PairFunction<Tuple2<Integer, Tuple3<Integer, Integer, double[]>>, Integer, Tuple2<Integer, double[]>> {

	private static final long serialVersionUID = 1L;
	private List<Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>>> model;

	public LabelClassification(
			List<Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>>> model) {
		this.model = model;
	}

	public Tuple2<Integer, Tuple2<Integer, double[]>> call(Tuple2<Integer, Tuple3<Integer, Integer, double[]>> t)
			throws Exception {
		int newId = -2;
		for (int i = 0; i < this.model.size(); i++) {
			// same subset
		//	System.out.println("Node: " + this.model.get(i)._2()._2()._3().intValue() + " node2: " + t._2()._1() + " object: " + t._2()._2());
			if (this.model.get(i)._2()._2()._3().intValue() == t._2()._1().intValue()) {
				for (int j = 0; j < this.model.get(i)._2()._2()._1().length; j++) {
					if (this.model.get(i)._2()._2()._1()[j] == t._1().intValue()) {
						newId = this.model.get(i)._2()._2()._2()[j];
					}
				}
			}
		}
		return new Tuple2<Integer, Tuple2<Integer, double[]>>(newId,
				new Tuple2<Integer, double[]>(t._2()._2(), t._2()._3()));
	}

}
