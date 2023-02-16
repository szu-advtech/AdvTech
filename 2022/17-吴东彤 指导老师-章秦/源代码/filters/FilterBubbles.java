package filters;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.CoreDistances;

public class FilterBubbles implements
		Function<Tuple2<String, CoreDistances>, Boolean> {

	private static final long serialVersionUID = 1L;

	public Boolean call(Tuple2<String, CoreDistances> v1) throws Exception {
		return v1._1().contains(" ");
	}

}
