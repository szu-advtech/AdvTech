package partition.mappers;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataPoints;
import distance.DistanceCalculator;
import distance.EuclideanDistance;

public class ClusterFeaturesByNodesMapper implements
		PairFunction<Tuple2<String, DataPoints>, String, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	Broadcast<double[][]> broadcastSampling;
	private int n;
	private double[] ls;
	private double[] ss;
	private Broadcast<int[][]> keysSamples; // use Accumulator


	public ClusterFeaturesByNodesMapper(Broadcast<double[][]> samples, Broadcast<int[][]> keysSamples) {
		this.broadcastSampling = samples;
		this.keysSamples = keysSamples;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<String, ClusterFeatureDataBubbles> call(Tuple2<String, DataPoints> t)
			throws Exception {
		double[] currentPoint = t._2.getPoint();
		int currentIdPoint = t._2.getIdPoint();
		String keyCurrentPoint = t._2.getKey();

		return new Tuple2(nearestNeighbors(currentPoint,
				broadcastSampling.value(), currentIdPoint, keyCurrentPoint).getKey(), 
				nearestNeighbors(currentPoint, broadcastSampling.value(), 
				currentIdPoint, keyCurrentPoint));
	}

	public ClusterFeatureDataBubbles nearestNeighbors(double[] currentPoint,
			double[][] sampling, int currentIdPoint,
			String keyCurrentPoint) {

		DistanceCalculator distanceFunction = new EuclideanDistance();

		// find nearest neighbor
		double nearestNeighbor = Double.MAX_VALUE;
		int index = 0;
		String[] nKeyPoint = keyCurrentPoint.split("\\.");
		int nodePoint = Integer.parseInt(nKeyPoint[0]);
		for (int neighbor = 0; neighbor < sampling.length; neighbor++) {
			if (nodePoint == this.keysSamples.value()[neighbor][0]) {
				double distance = distanceFunction.computeDistance(sampling[neighbor], currentPoint);
				if (distance < nearestNeighbor) {
					nearestNeighbor = distance;
					index = neighbor;
				}
			}
		}
		n = 1;
		this.ls = new double[sampling[0].length];
		this.ss = new double[sampling[0].length];
		for (int j = 0; j < ls.length; j++) {
			ls[j] = currentPoint[j];
			ss[j] = (currentPoint[j] * currentPoint[j]);
		}

		StringBuilder string = new StringBuilder();
		string.append(nKeyPoint[0] + "." + index + "." + nKeyPoint[2]);
		String key = string.toString();
		return new ClusterFeatureDataBubbles(key, index, ls, ss, n,
				ls/* esse é o rep único */, 0, 0, new DataPoints(currentIdPoint, index, 0, key, currentPoint), 0);
	}

	
	public double[] getLs() {
		return ls;
	}

	public void setLs(double[] ls) {
		this.ls = ls;
	}

	public double[] getSs() {
		return ss;
	}

	public void setSs(double[] ss) {
		this.ss = ss;
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}

	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public Broadcast<double[][]> getBroadcastSampling() {
		return broadcastSampling;
	}

	public void setBroadcastSampling(Broadcast<double[][]> broadcastSampling) {
		this.broadcastSampling = broadcastSampling;
	}

	public Broadcast<int[][]> getKeysSamples() {
		return keysSamples;
	}

	public void setKeysSamples(Broadcast<int[][]> keysSamples) {
		this.keysSamples = keysSamples;
	}
}
