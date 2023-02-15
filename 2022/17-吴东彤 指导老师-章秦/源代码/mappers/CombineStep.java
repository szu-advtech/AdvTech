package mappers;

import org.apache.spark.api.java.function.Function2;

import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

public class CombineStep implements
		Function2<Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> {

	private static final long serialVersionUID = 1L;

	public CombineStep() {

	}

	public Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer> call(
			Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer> partialBubble1,
			Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer> partialBubble2) throws Exception {
		double[] rep;
		double[] ls = new double[partialBubble1._2()._2().length];
		double[] ss = new double[partialBubble1._2()._2().length];
		for (int i = 0; i < partialBubble1._2()._2().length; i++) {
			ls[i] = partialBubble1._2()._2()[i] + partialBubble2._2()._2()[i]; // LS
			ss[i] = partialBubble1._2()._3()[i] + partialBubble2._2()._3()[i]; // SS
		}
		partialBubble1._2()._4()[2] += 1; // n
		// parameters ls, n
		rep = computeRepBubble(ls, partialBubble1._2()._4()[2]);
		// parameters - LS, SS, n
		partialBubble1._2()._4()[0] = computeExtentBubble(ls, ss, partialBubble1._2()._4()[2]);
		// parameters - extent, n, numberOfAttributes;
		partialBubble1._2()._4()[1] = computeNNDistBubble(partialBubble1._2()._4()[0], partialBubble1._2()._4()[2],
				ls.length, 1);
		return new Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>(partialBubble1._1(),
				// rep, ls, ss, (extent, NNDist, n)
				new Tuple4<double[], double[], double[], double[]>(rep, ls, ss, partialBubble1._2()._4()),
				partialBubble1._3());
	}

	private double computeNNDistBubble(double extent, double n, int numberOfAttributes, int k) {
		return Math.pow((k / n), (1 / numberOfAttributes)) * extent;
	}

	private double computeExtentBubble(double[] ls, double[] ss, double n) {
		double extent = 0.0;
		if (n > 1) {
			for (int i = 0; i < ls.length; i++) {
				if (((2 * n * ss[i]) - (2 * (ls[i] * ls[i]))) >= 0) {
					extent += Math.sqrt(((2 * n * ss[i]) - (2 * (ls[i] * ls[i]))) / (n * (n - 1)));
				}
			}
		}
		return extent / ls.length;
	}

	private double[] computeRepBubble(double[] ls, double n) {
		double[] rep = new double[ls.length];
		for (int i = 0; i < ls.length; i++) {
			rep[i] = ls[i] / n;
		}
		return rep;
	}

}
