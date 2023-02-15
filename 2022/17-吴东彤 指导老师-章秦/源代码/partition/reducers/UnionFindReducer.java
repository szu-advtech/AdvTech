package partition.reducers;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.api.java.function.Function2;

import datastructure.MinimumSpanningTree;
import datastructure.SortMST;

public class UnionFindReducer implements Function2<String, String, String> {
	private static final long serialVersionUID = 1L;

	public UnionFindReducer() {
		
	}

	public String call(String v1, String v2) throws Exception {

		LinkedList<MinimumSpanningTree> list = new LinkedList<MinimumSpanningTree>();
		String[] str1 = v1.split("\n");
		String[] str2 = v2.split("\n");

		for (int i = 0; i < str1.length; i++) {
			String[] data = str1[i].split(" ");
			int vertice1 = Integer.parseInt(data[0]);
			int vertice2 = Integer.parseInt(data[1]);
			double weight = Double.parseDouble(data[2]);
			int fake1 = Integer.parseInt(data[3]);
			int fake2 = Integer.parseInt(data[4]);
			int node = Integer.parseInt(data[5]);
			list.add(new MinimumSpanningTree(vertice1, vertice2, weight, fake1,
					fake2, node));
		}
		for (int i = 0; i < str2.length; i++) {
			String[] data = str2[i].split(" ");
			int vertice1 = Integer.parseInt(data[0]);
			int vertice2 = Integer.parseInt(data[1]);
			double weight = Double.parseDouble(data[2]);
			int fake1 = Integer.parseInt(data[3]);
			int fake2 = Integer.parseInt(data[4]);
			int node = Integer.parseInt(data[5]);
			list.add(new MinimumSpanningTree(vertice1, vertice2, weight, fake1,
					fake2, node));
		}
		SortMST sort = new SortMST();
		Collections.sort(list, sort);

		StringBuilder concat = new StringBuilder();
		for (int i = 0; i < list.size() - 1; i++) {
			concat.append(list.get(i).getVertice1() + " "
					+ list.get(i).getVertice2() + " " + list.get(i).getWeight()
					+ " " + list.get(i).getFake1() + " "
					+ list.get(i).getFake2() + " " + list.get(i).getNode()
					+ "\n");
		}

		concat.append(list.get(list.size() - 1).getVertice1() + " "
				+ list.get(list.size() - 1).getVertice2() + " "
				+ list.get(list.size() - 1).getWeight() + " "
				+ list.get(list.size() - 1).getFake1() + " "
				+ list.get(list.size() - 1).getFake2() + " "
				+ list.get(list.size() - 1).getNode());

		list = null;

		return concat.toString();
	}

	public int checkId(int vertice, List<Integer> list) {
		for (int i = 0; i < list.size(); i++) {
			if (vertice == list.get(i)) {
				return i;
			}
		}
		return 0;
	}

}
