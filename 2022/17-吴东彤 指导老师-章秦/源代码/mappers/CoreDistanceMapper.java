package mappers;

import java.io.Serializable;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;
import datastructure.CoreDistances;
import datastructure.DataPoints;
import distance.DistanceCalculator;

public class CoreDistanceMapper implements
		PairFunction<Tuple2<Integer, DataPoints>, Integer, CoreDistances>, Serializable {

	private static final long serialVersionUID = 1L;
	private Broadcast<double[][]> dataset;
	private Broadcast<Integer> mpts;
	private Broadcast<DistanceCalculator> distance;

	public CoreDistanceMapper(Broadcast<double[][]> dataset, Broadcast<Integer> mpts,
			Broadcast<DistanceCalculator> distance) {
		this.setDataset(dataset);
		this.setMpts(mpts);
		this.setDistance(distance);
	}
	
	public Broadcast<double[][]> getDataset() {
		return dataset;
	}

	public void setDataset(Broadcast<double[][]> dataset) {
		this.dataset = dataset;
	}

	public Broadcast<Integer> getMpts() {
		return mpts;
	}

	public void setMpts(Broadcast<Integer> mpts) {
		this.mpts = mpts;
	}

	public Broadcast<DistanceCalculator> getDistance() {
		return distance;
	}

	public void setDistance(Broadcast<DistanceCalculator> distance) {
		this.distance = distance;
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Tuple2<Integer, CoreDistances> call(Tuple2<Integer, DataPoints> t)
			throws Exception {
		int pointID = t._2.getIdPoint();
		int tmpIdPoint = t._2.getTmpIdPoint();
		double[] point = t._2.getPoint();
		double coreDistance = calculateCoreDistances(dataset.getValue(), point, mpts.getValue(), distance.getValue());
		CoreDistances core = new CoreDistances();
		core.setCoreDistance(coreDistance);
		core.setIdPoint(pointID);
		core.setTmpIdPoint(tmpIdPoint);
		core.setPoint(point);
		return new Tuple2(pointID, core);
	}

	public static double calculateCoreDistances(double[][] dataSet,
			double[] currentPoint, int k, DistanceCalculator distanceFunction) {
		int numNeighbors = k - 1;
		double coreDistance = 0;

		if (k == 1) {
			return coreDistance;
		}

		double[] kNNDistances = new double[numNeighbors];
		for (int i = 0; i < numNeighbors; i++) {
			kNNDistances[i] = Double.MAX_VALUE;
		}

		for (int neighbor = 0; neighbor < dataSet.length; neighbor++) {
			double distance = distanceFunction.computeDistance(currentPoint,
					dataSet[neighbor]);

			// Check at which position in the nearest distances the current
			// distance would fit:
			int neighborIndex = numNeighbors;
			while (neighborIndex >= 1
					&& distance < kNNDistances[neighborIndex - 1]) {
				neighborIndex--;
			}

			// Shift elements in the array to make room for the current
			// distance:
			if (neighborIndex < numNeighbors) {
				for (int shiftIndex = numNeighbors - 1; shiftIndex > neighborIndex; shiftIndex--) {
					kNNDistances[shiftIndex] = kNNDistances[shiftIndex - 1];
				}
				kNNDistances[neighborIndex] = distance;
			}
		}
		coreDistance = kNNDistances[numNeighbors - 1];

		return coreDistance;
	}
}
