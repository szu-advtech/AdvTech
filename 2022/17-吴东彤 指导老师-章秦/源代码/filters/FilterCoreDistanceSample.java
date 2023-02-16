package filters;

import org.apache.spark.api.java.function.Function;


import scala.Tuple2;
import datastructure.CoreDistances;

public class FilterCoreDistanceSample implements
		Function<Tuple2<Integer, CoreDistances>, Boolean> {

	private static final long serialVersionUID = 1L;
	private int value;

	public FilterCoreDistanceSample(int value) {
		this.value = value;
	}

	public Boolean call(Tuple2<Integer, CoreDistances> v1) throws Exception {
		return v1._2.getIdPoint() == this.value;
	}
}
