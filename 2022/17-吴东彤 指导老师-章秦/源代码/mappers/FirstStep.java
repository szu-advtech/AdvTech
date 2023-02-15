package mappers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import distance.DistanceCalculator;
import hdbscanstar.HDBSCANStar;
import hdbscanstar.UndirectedGraph;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

public class FirstStep implements
		PairFlatMapFunction<Iterator<Tuple2<Integer, Tuple2<Integer, double[]>>>, Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> {

	private static final long serialVersionUID = 1L;
	private int mpts;
	private double k;
	private String inputFile;
	private int processingUnits;
	private int iteration;
	private DistanceCalculator distanceFunction;
	private List<Tuple2<Integer, Tuple2<Integer, double[]>>> bSamples;
	private List<Tuple2<Integer, Tuple2<Integer, Integer>>> keyCounts;

	public FirstStep(Double k, Integer processingUnits, Integer mpts, DistanceCalculator distanceFunction,
			String inputFile, int iteration, List<Tuple2<Integer, Tuple2<Integer, double[]>>> bSamples,
			List<Tuple2<Integer, Tuple2<Integer, Integer>>> keyCounts) {
		this.distanceFunction = distanceFunction;
		this.mpts = mpts;
		this.k = k;
		this.inputFile = inputFile;
		this.processingUnits = processingUnits;
		this.iteration = iteration;
		this.bSamples = bSamples;
		this.keyCounts = keyCounts;
	}

	public Iterator<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>> call(
			Iterator<Tuple2<Integer, Tuple2<Integer, double[]>>> t) throws Exception {
		ArrayList<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>> subset = new ArrayList<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>>();
		boolean computeItOnce = false;
		HDBSCANStar model = new HDBSCANStar();
		int rows = 0;
		double[][] dataset = null;
		int[] indices = null;
        //int count = 0;
        Tuple2<Integer, Integer> type = null;
		while (t.hasNext()) {
			Tuple2<Integer, Tuple2<Integer, double[]>> record = t.next();
			if (computeItOnce == false) {
				for (Tuple2<Integer, Tuple2<Integer, Integer>> element : this.keyCounts) {
					if (element._1() == record._1().intValue()) {
						type = new Tuple2<Integer, Integer>(record._1(), element._2()._1());
						computeItOnce = true;
						if(element._2()._1().intValue() <= this.processingUnits){
							dataset = new double[type._2().intValue()][record._2()._2().length];
							indices = new int[type._2().intValue()];
						}
					}
				}
			}
			if (type._2() <= this.processingUnits) {
				// collect data to compute MST ***
				//System.out.println("rows="+rows+",dataset_size="+dataset.length);
				dataset[rows] = record._2()._2();
				indices[rows] = record._2()._1().intValue();
				rows++;
			} else { // compute the nearest neighbor of the first object
				double minDistance = Double.MAX_VALUE;
				int nearestNeighbor = 0;
				for (int neighbor = 0; neighbor < this.bSamples.size(); neighbor++) {
					// compute distances
					double dist = this.distanceFunction.computeDistance(record._2()._2(),
							this.bSamples.get(neighbor)._2()._2());
					if (dist < minDistance) {
						minDistance = dist;
						nearestNeighbor = this.bSamples.get(neighbor)._2()._1().intValue();
					}
				}
				// idSample, idObject, object, id(subset)
				double[] infoBubbles = new double[3];
				infoBubbles[2] = 1;
				double[] ss = new double[record._2()._2().length];
				double[] ls = new double[record._2()._2().length];
				for (int i = 0; i < ss.length; i++) {
					ss[i] = record._2()._2()[i] * record._2()._2()[i];
					ls[i] = record._2()._2()[i];
				}
				subset.add(
						new Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>(
								nearestNeighbor,
								new Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>(
										record._2()._1(),
										new Tuple4<double[], double[], double[], double[]>(ls, ls, ss, infoBubbles),
										record._1())));
			}
		}
		if ((dataset != null) && (type._2() <= this.processingUnits)) { // compute local MST from subset
			double[] coreDistances = model.calculateCoreDistances(dataset, this.mpts, this.distanceFunction);
			UndirectedGraph mst = model.constructMST(dataset, coreDistances, true, distanceFunction, indices,
					indices.length);
			for (int i = 0; i < mst.getNumEdges(); i++) {
				// 1, v, dmreach, u
				double[] dmreach = new double[1];
				dmreach[0] = mst.getEdgeWeightAtIndex(i);
				subset.add(
						new Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>(
								-1,
								new Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>(
										mst.getFirstVertexAtIndex(i),
										new Tuple4<double[], double[], double[], double[]>(dmreach, null, null, null),
										mst.getSecondVertexAtIndex(i))));
			}
		}
		return subset.iterator();
	}
}
