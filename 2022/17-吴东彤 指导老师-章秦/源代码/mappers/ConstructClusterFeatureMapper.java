package mappers;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataPoints;
import distance.DistanceCalculator;
import distance.EuclideanDistance;

public class ConstructClusterFeatureMapper
		implements
		PairFunction<Tuple2<String, DataPoints>, Integer, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	Broadcast<double[][]> broadcastSampling;
	Broadcast<String[]> bKeySampling;
	private int n;
	private double[] ls;
	private double[] ss;

	public ConstructClusterFeatureMapper() {
	}

	public ConstructClusterFeatureMapper(Broadcast<double[][]> b) {
		this.broadcastSampling = b;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<Integer, ClusterFeatureDataBubbles> call(
			Tuple2<String, DataPoints> t) throws Exception {
		double[] currentPoint = t._2.getPoint();
		int currentIdPoint = t._2.getIdPoint();

		return new Tuple2(nearestNeighbors(currentPoint,
				broadcastSampling.value(), broadcastSampling.value()[0].length,
				currentIdPoint).getId(), nearestNeighbors(currentPoint,
				broadcastSampling.value(), broadcastSampling.value()[0].length,
				currentIdPoint));
	}

	public ClusterFeatureDataBubbles nearestNeighbors(double[] currentPoint,
			double[][] sampling, int col, int currentIdPoint) {

		DistanceCalculator distanceFunction = new EuclideanDistance();

		// find nearest neighbor
		double nearestNeighbor = Double.MAX_VALUE;
		int index = 0;

		for (int neighbor = 0; neighbor < sampling.length; neighbor++) {
			double distance = distanceFunction.computeDistance(
					sampling[neighbor], currentPoint);
			if (distance < nearestNeighbor) {
				nearestNeighbor = distance;
				index = neighbor;
			}
		}
		n = 1;
		this.ls = new double[broadcastSampling.value()[0].length];
		this.ss = new double[broadcastSampling.value()[0].length];
		for (int j = 0; j < sampling[0].length; j++) {
			ls[j] = currentPoint[j];
			ss[j] = (currentPoint[j] * currentPoint[j]);
		}

		return new ClusterFeatureDataBubbles(index, ls, ss, n, ls /* rep */, 0,
				0, new DataPoints(currentIdPoint, index, 0, currentPoint), 0);
	}

	// / getters and setters methods

	public Broadcast<double[][]> getBroadcastSampling() {
		return broadcastSampling;
	}

	public void setBroadcastSampling(Broadcast<double[][]> broadcastSampling) {
		this.broadcastSampling = broadcastSampling;
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
}
