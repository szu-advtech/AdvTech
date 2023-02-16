package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;

public class findConnectedComponentsOnMST implements PairFlatMapFunction<Tuple2<Integer, int[]>, Integer, int[]> {

	private static final long serialVersionUID = 1L;

	public Iterator<Tuple2<Integer, int[]>> call(Tuple2<Integer, int[]> t) throws Exception {
		Arrays.sort(t._2());
		ArrayList<Tuple2<Integer, int[]>> setEdges = new ArrayList<Tuple2<Integer, int[]>>();
		boolean isLocMaxState = false;
		int vFirst = t._2()[0];
		int vSource = t._1();
		Map<Integer, ArrayList<Integer>> edges = new HashMap<Integer, ArrayList<Integer>>();
		if (vSource <= vFirst) {
			isLocMaxState = true;
			if (edges.get(vSource) == null) {
				edges.put(vSource, new ArrayList<Integer>());
			}
			edges.get(vSource).add(vFirst);
		}
		int vDestOld = vFirst;
		int vDest = vFirst;
		for (Integer adj : t._2()) {
			//System.out.print(" " + adj);
			vDest = adj;
			if (vDest == vDestOld) {
				continue;
			}
			if (isLocMaxState) {
				edges.get(vSource).add(vDest);
			} else {
				if (edges.get(vFirst) == null) {
					edges.put(vFirst, new ArrayList<Integer>());
				}
				if (edges.get(vDest) == null) {
					edges.put(vDest, new ArrayList<Integer>());
				}
				edges.get(vFirst).add(vDest);
				edges.get(vDest).add(vFirst);
				Main.newIteration.add(1);
			}
			vDestOld = vDest;
		}
		//System.out.println();
		if (vSource < vDest && !isLocMaxState) {
			if (edges.get(vSource) == null) {
				edges.put(vSource, new ArrayList<Integer>());
			}
			edges.get(vSource).add(vFirst);
		}
		for (Integer key : edges.keySet()) {
			int[] adjVertice = new int[edges.get(key).size()];
			Iterator<Integer> iterator = edges.get(key).iterator();
			int count = 0;
			while (iterator.hasNext()) {
				adjVertice[count] = iterator.next().intValue();
				count++;
			}
			setEdges.add(new Tuple2<Integer, int[]>(key, adjVertice));
		}
		return setEdges.iterator();
	}

}
