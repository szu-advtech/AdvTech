package mappers;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;
import datastructure.CoreDistances;

public class ChangeKey implements
		Function<Tuple2<String, CoreDistances>, String> {

	private static final long serialVersionUID = 1L;

	public String call(Tuple2<String, CoreDistances> t) throws Exception {
		StringBuilder string = new StringBuilder();
		for (int i = 0; i < t._2().getMst().size() - 1; i++) {
			string.append(t._2().getMst().get(i).getVertice1() + " "
					+ t._2().getMst().get(i).getVertice2() + " "
					+ t._2().getMst().get(i).getWeight() + " "
					+ t._2().getMst().get(i).getVertice1() + " "
					+ t._2().getMst().get(i).getVertice2()
					+ t._2().getMst().get(i).getNode() + "\n");
		}
		string.append(t._2().getMst().get(t._2().getMst().size() - 1)
				.getVertice1()
				+ " "
				+ t._2().getMst().get(t._2().getMst().size() - 1).getVertice2()
				+ " "
				+ t._2().getMst().get(t._2().getMst().size() - 1).getWeight()
				+ " "
				+ t._2().getMst().get(t._2().getMst().size() - 1).getVertice1()
				+ " "
				+ t._2().getMst().get(t._2().getMst().size() - 1).getVertice2()
				+ " "
				+ t._2().getMst().get(t._2().getMst().size() - 1).getNode());

		return string.toString();
	}
}
