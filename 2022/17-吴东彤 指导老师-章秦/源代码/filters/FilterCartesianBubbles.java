package filters;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.CoreDistances;

public class FilterCartesianBubbles implements
		Function<Tuple2<String, CoreDistances>, Boolean> {

	private static final long serialVersionUID = 1L;

	public Boolean call(Tuple2<String, CoreDistances> v1) throws Exception {
		String[] keys = v1._1().split(" ");
		int[] sArray = new int[keys.length];
		   for(int i = 0; i < sArray.length; i++){
			   sArray[i] = Integer.parseInt(keys[i]);
		   }
		return sArray[0] < sArray[1];
	}

}
