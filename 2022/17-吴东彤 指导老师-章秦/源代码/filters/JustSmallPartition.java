package filters;

import org.apache.spark.api.java.function.Function;

import datastructure.DataPoints;
import scala.Tuple2;

public class JustSmallPartition
		implements Function<Tuple2<String, DataPoints>, Boolean> {

	private static final long serialVersionUID = 1L;
	private int process;

	public JustSmallPartition(int process) {
		this.setProcess(process);
	}

	public Boolean call(Tuple2<String, DataPoints> v1) throws Exception {
		String[] key = v1._2().getKey().split("\\.");
		int size = Integer.parseInt(key[2]);
		return size <= this.getProcess();
	}

	public int getProcess() {
		return process;
	}

	public void setProcess(int process) {
		this.process = process;
	}
}
