package main;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

public class ReMapBubbles implements
        // idBubble, idObject, 
		PairFunction<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>, Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>> {

	private static final long serialVersionUID = 1L;

	public Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>> call(
			Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> t)
			throws Exception {
		// id, bubbles, partition from bubbles, inter-cluster edges;
		double[][] infoB1 = new double[1][t._2()._2()._1().length];
		double[][] infoB2 = new double[1][t._2()._2()._4().length];
		int[] ids = new int[1];
		infoB1[0] = t._2()._2()._1();
		infoB2[0] = t._2()._2()._4();
		ids[0] = t._1().intValue();
		return new Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>>(
				t._2()._3(),
				new Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>(
						new Tuple3<double[][], double[][], int[]>(infoB1, infoB2, ids),
						new Tuple3<int[], int[], Integer>(null, null, t._2()._3()),
						new Tuple3<int[], int[], double[]>(null, null, null)));
	}

}
