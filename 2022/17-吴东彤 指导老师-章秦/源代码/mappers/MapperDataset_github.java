package mappers;

import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

public class MapperDataset_github implements PairFunction<String, Integer, Tuple2<Integer, double[]>> {

	private static final long serialVersionUID = 1L;
	private int count = -1;

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<Integer, Tuple2<Integer, double[]>> call(String s) throws Exception {
		String[] sarray = s.split(" ");
		double[] dataPoint = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {

			dataPoint[i] = Double.parseDouble(sarray[i]);
		}
		count++;
		return new Tuple2(0, new Tuple2(count, dataPoint));
	}
}