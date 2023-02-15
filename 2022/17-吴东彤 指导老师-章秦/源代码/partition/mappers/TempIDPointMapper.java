package partition.mappers;

import java.util.LinkedList;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.DataPoints;

public class TempIDPointMapper implements
		PairFunction<Tuple2<String, DataPoints>, String, DataPoints> {

	private static final long serialVersionUID = 1L;
    private int newId;

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, DataPoints> call(Tuple2<String, DataPoints> t)
			throws Exception {
        LinkedList<DataPoints> dataPoints = new LinkedList<DataPoints>();
        dataPoints.add(t._2());
              
		return new Tuple2(t._1(), new DataPoints(t._2.getIdPoint(),
				this.newId++, t._2.getIdBubble(), t._2().getKey(),
				t._2.getPoint(), dataPoints));
	}
}
