package filters;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.CompleteGraph;

public class TopDiagonalFilter implements
		Function<Tuple2<Double, CompleteGraph>, Boolean> {

	private static final long serialVersionUID = 1L;

	public Boolean call(Tuple2<Double, CompleteGraph> v1) throws Exception {
		return v1._2.getVertex1() < v1._2.getVertex2();
	}
}
