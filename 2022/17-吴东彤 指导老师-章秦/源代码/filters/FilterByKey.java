package filters;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;


import scala.Tuple2;
import datastructure.DataPoints;

public class FilterByKey implements Function<Tuple2<Integer, DataPoints>, Boolean> {
	private static final long serialVersionUID = 1L;
	private int indice;

	public FilterByKey() {
	}

	public FilterByKey(int indice) {
		this.setIndice(indice);
	}
	public Boolean call(Tuple2<Integer, DataPoints> clusters) throws Exception {
		return clusters._1().intValue() == this.getIndice();
	}

	public int getIndice() {
		return this.indice;
	}

	public void setIndice(int indice) {
		this.indice = indice;
	}
}
