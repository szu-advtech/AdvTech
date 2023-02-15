package partition.mappers;

import hdbscanstar.UndirectedGraph;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;
import datastructure.DataPoints;
import distance.DistanceCalculator;

public class LocalMSTMapperPartition
		implements
		PairFlatMapFunction<Iterator<Tuple2<String, DataPoints>>, String, DataPoints> {

	private static final long serialVersionUID = 1L;
	private Integer minPoints;
	private DistanceCalculator distanceFunction;
	private LinkedList<LinkedList<DataPoints>> listOfList;

	public LocalMSTMapperPartition(Integer minPoints,
			DistanceCalculator distanceFunction,
			LinkedList<LinkedList<DataPoints>> listOfList) {
		this.setMinPoints(minPoints);
		this.setDistanceFunction(distanceFunction);
		this.setListOfList(listOfList);
	}

	@SuppressWarnings("unchecked")
	public Iterator<Tuple2<String, DataPoints>> call(
			Iterator<Tuple2<String, DataPoints>> t) throws Exception {
		Tuple2<String, DataPoints> tuple = null;
		while (t.hasNext()) {
			tuple = t.next();
			String keyCurrentPoint = tuple._2().getKey();

			for (int i = 0; i < this.listOfList.size(); i++) {
				LinkedList<DataPoints> list = this.listOfList.get(i);
				if (list.get(0).getKey().equals(keyCurrentPoint)) {
					// create msts
					double[] coreDistances = calculateCoreDistances(list,
							this.minPoints, this.distanceFunction);
					UndirectedGraph mst = constructMST(list, coreDistances,
							true, this.distanceFunction);
					mst.quicksortByEdgeWeight();

					DataPoints localMST = new DataPoints(mst);
					LinkedList<DataPoints> listLocalMST = new LinkedList<DataPoints>();
					listLocalMST.add(localMST);
					tuple = new Tuple2<String, DataPoints>("mst",
							new DataPoints(listLocalMST)); // uma
															// linkedList<UndirectedGraph>
				}
			}
		}
		return Arrays.asList(tuple).iterator();
	}

	public double[] calculateCoreDistances(LinkedList<DataPoints> dataset,
			int k, DistanceCalculator distanceFunction) {
		int numNeighbors = k - 1;
		double[] coreDistances = new double[dataset.size()];

		if (k == 1) {
			for (int point = 0; point < dataset.size(); point++) {
				coreDistances[point] = 0;
			}
			return coreDistances;
		}

		for (int point = 0; point < dataset.size(); point++) {
			double[] kNNDistances = new double[numNeighbors]; // Sorted nearest
																// distances
																// found so far
			for (int i = 0; i < numNeighbors; i++) {
				kNNDistances[i] = Double.MAX_VALUE;
			}

			for (int neighbor = 0; neighbor < dataset.size(); neighbor++) {
				if (point == neighbor)
					continue;
				double distance = distanceFunction.computeDistance(
						dataset.get(point).getPoint(), dataset.get(neighbor)
								.getPoint());

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
			coreDistances[point] = kNNDistances[numNeighbors - 1];
		}
		return coreDistances;
	}

	public UndirectedGraph constructMST(LinkedList<DataPoints> dataset,
			double[] coreDistances, boolean selfEdges,
			DistanceCalculator distanceFunction) {

		int selfEdgeCapacity = 0;
		if (selfEdges)
			selfEdgeCapacity = dataset.size();

		// One bit is set (true) for each attached point, or unset (false) for
		// unattached points:
		BitSet attachedPoints = new BitSet(dataset.size());

		// Each point has a current neighbor point in the tree, and a current
		// nearest distance:
		int[] nearestMRDNeighbors = new int[dataset.size() - 1
				+ selfEdgeCapacity];
		double[] nearestMRDDistances = new double[dataset.size() - 1
				+ selfEdgeCapacity];

		for (int i = 0; i < dataset.size() - 1; i++) {
			nearestMRDDistances[i] = Double.MAX_VALUE;
		}

		// The MST is expanded starting with the last point in the data set:
		int currentPoint = dataset.size() - 1;
		int numAttachedPoints = 1;
		attachedPoints.set(dataset.size() - 1);

		// Continue attaching points to the MST until all points are attached:
		while (numAttachedPoints < dataset.size()) {
			int nearestMRDPoint = -1;
			double nearestMRDDistance = Double.MAX_VALUE;

			// Iterate through all unattached points, updating distances using
			// the current point:
			for (int neighbor = 0; neighbor < dataset.size(); neighbor++) {
				if (currentPoint == neighbor)
					continue;
				if (attachedPoints.get(neighbor) == true)
					continue;

				double distance = distanceFunction.computeDistance(
						dataset.get(currentPoint).getPoint(),
						dataset.get(neighbor).getPoint());

				double mutualReachabiltiyDistance = distance;
				if (coreDistances[currentPoint] > mutualReachabiltiyDistance)
					mutualReachabiltiyDistance = coreDistances[currentPoint];
				if (coreDistances[neighbor] > mutualReachabiltiyDistance)
					mutualReachabiltiyDistance = coreDistances[neighbor];

				if (mutualReachabiltiyDistance < nearestMRDDistances[neighbor]) {
					nearestMRDDistances[neighbor] = mutualReachabiltiyDistance;
					nearestMRDNeighbors[neighbor] = currentPoint;
				}

				// Check if the unattached point being updated is the closest to
				// the tree:
				if (nearestMRDDistances[neighbor] <= nearestMRDDistance) {
					nearestMRDDistance = nearestMRDDistances[neighbor];
					nearestMRDPoint = neighbor;
				}
			}

			// Attach the closest point found in this iteration to the tree:
			attachedPoints.set(nearestMRDPoint);
			numAttachedPoints++;
			currentPoint = nearestMRDPoint;
		}

		// Create an array for vertices in the tree that each point attached to:
		int[] otherVertexIndices = new int[dataset.size() - 1
				+ selfEdgeCapacity];
		for (int i = 0; i < dataset.size() - 1; i++) {
			otherVertexIndices[i] = i;
		}

		// If necessary, attach self edges:
		if (selfEdges) {
			for (int i = dataset.size() - 1; i < dataset.size() * 2 - 1; i++) {
				int vertex = i - (dataset.size() - 1);
				nearestMRDNeighbors[i] = vertex;
				otherVertexIndices[i] = vertex;
				nearestMRDDistances[i] = coreDistances[vertex];
			}
		}

		return new UndirectedGraph(dataset.size(), nearestMRDNeighbors,
				otherVertexIndices, nearestMRDDistances);
	}

	public LinkedList<LinkedList<DataPoints>> getListOfList() {
		return listOfList;
	}

	public void setListOfList(LinkedList<LinkedList<DataPoints>> listOfList) {
		this.listOfList = listOfList;
	}

	public Integer getMinPoints() {
		return minPoints;
	}

	public void setMinPoints(Integer minPoints) {
		this.minPoints = minPoints;
	}

	public DistanceCalculator getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(DistanceCalculator distanceFunction) {
		this.distanceFunction = distanceFunction;
	}
}
