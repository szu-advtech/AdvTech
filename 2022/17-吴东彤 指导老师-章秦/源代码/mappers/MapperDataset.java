package mappers;

import org.apache.spark.api.java.function.PairFunction;

import datastructure.DataPoints;

import scala.Tuple2;

public class MapperDataset implements PairFunction<String, String, DataPoints> {

	private static final long serialVersionUID = 1L;
	private int count = -1;

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, DataPoints> call(String s) throws Exception {
		String[] sarray = s.split(" ");
		double[] dataPoint = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
			dataPoint[i] = Double.parseDouble(sarray[i]);
		}
		count++;
		StringBuilder key = new StringBuilder();
		key.append(0 + "." + 0 + "." + 0);
		// return new Tuple2(0, new DataPoints(count, 0, 0, dataPoint)); //
		// antigo
		return new Tuple2(0, new DataPoints(count, 0, 0, 0, key.toString(),
				dataPoint));
	}

}
