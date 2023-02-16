package partition.filters;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.DataPoints;

public class FilterFakeMST implements
		Function<Tuple2<String, DataPoints>, Boolean> {

	private static final long serialVersionUID = 1L;

	public Boolean call(Tuple2<String, DataPoints> v1) throws Exception {
		return !v1._1().contains("_");
	}

}
