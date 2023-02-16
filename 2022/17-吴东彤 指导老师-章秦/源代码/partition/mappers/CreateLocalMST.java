package partition.mappers;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Tuple2;
import datastructure.DataPoints;
import datastructure.MinimumSpanningTree;
//import datastructure.SortMST;
import distance.DistanceCalculator;

public class CreateLocalMST implements
		FlatMapFunction<Iterator<Tuple2<String, DataPoints>>, String> {

	private static final long serialVersionUID = 1L;
	private Integer k;
	private Integer processU;
	private Integer minPoints;
	private DistanceCalculator distanceFunction;
	private String inputFile;
	private int countLevels;

	public CreateLocalMST(Integer k, Integer processing_units,
			Integer minPoints, DistanceCalculator distanceFunction,
			String inputFile, int countLevels) {
		this.setK(k);
		this.setProcessU(processing_units);
		this.setMinPoints(minPoints);
		this.setDistanceFunction(distanceFunction);
		this.setInputFile(inputFile);
		this.setCountLevels(countLevels);
	}

	public Iterator<String> call(Iterator<Tuple2<String, DataPoints>> t)
			throws Exception {
		LinkedList<DataPoints> dataset = new LinkedList<DataPoints>();

		StringBuilder mstResult = new StringBuilder();

		while (t.hasNext()) {
			dataset.add(t.next()._2);
		}

		double[][] data = null;
		String[] keys = null;
		int[] indices = null;
		if ((dataset.size() <= this.processU) && !dataset.isEmpty()) {
			data = new double[dataset.size()][dataset.get(0).getPoint().length];
			keys = new String[dataset.size()];
			indices = new int[dataset.size()];
			for (int i = 0; i < data.length; i++) {
				for (int j = 0; j < data[0].length; j++) {
					data[i][j] = dataset.get(i).getPoint()[j];
				}
				keys[i] = dataset.get(i).getKey();
				indices[i] = dataset.get(i).getIdPoint();
			}
		}

		dataset = null;

		if (data != null) {
			String[] key = keys[0].split("\\.");
			// int size = Integer.parseInt(key[2]);
			int node = Integer.parseInt(key[0]);
			// if (data.length > this.processU) {
			// //tuple = new Tuple2<String, String>("clusters", "");
			// }
			// computar mst
			// else
			if (data.length <= this.processU) {
				// samples = dataset;
				// create msts
				double[] coreDistances = calculateCoreDistances(data,
						this.minPoints, this.distanceFunction);
				MinimumSpanningTree mst = constructMST(data, indices,
						coreDistances, true, this.distanceFunction, node);

				// sort mst
				//Collections.sort(mst.getTree(), new SortMST());
				// //////////

				// Configuration configuration = new Configuration();
				// configuration.addResource(
				// new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
				// configuration.addResource(
				// new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));
				// FileSystem fs = FileSystem.get(configuration);

				// melhor escrever o objeto UndirectedGraph, depois ler e
				// jogar numa LinkedList e ordenar em um reduce

				// Path pathLocalMst = new Path(
				// this.getInputFile() + "/resultados/mst/" + countLevels
				// + "_" + node + "/mst"+node);
				// BufferedWriter mstWriter = new BufferedWriter(
				// new OutputStreamWriter(fs.create(pathLocalMst, true)));
				for (int i = 0; i < mst.getTree().size() - 1; i++) {
					// mstWriter.write(mst.getTree().get(i).getVertice1() + " "
					// + mst.getTree().get(i).getVertice2() + " "
					// + mst.getTree().get(i).getWeight() + " "
					// + mst.getTree().get(i).getFake1() + " "
					// + mst.getTree().get(i).getFake2() + " "
					// + mst.getTree().get(i).getNode() + "\n");

					mstResult.append(mst.getTree().get(i).getVertice1() + " "
							+ mst.getTree().get(i).getVertice2() + " "
							+ mst.getTree().get(i).getWeight() + " "
							+ mst.getTree().get(i).getFake1() + " "
							+ mst.getTree().get(i).getFake2() + " "
							+ mst.getTree().get(i).getNode() + "\n");
				}
				
				mstResult.append(mst.getTree().get(mst.getTree().size() - 1).getVertice1() + " "
						+ mst.getTree().get(mst.getTree().size() - 1).getVertice2() + " "
						+ mst.getTree().get(mst.getTree().size() - 1).getWeight() + " "
						+ mst.getTree().get(mst.getTree().size() - 1).getFake1() + " "
						+ mst.getTree().get(mst.getTree().size() - 1).getFake2() + " "
						+ mst.getTree().get(mst.getTree().size() - 1).getNode());
				
				// mstWriter.close();
				// tuple = new Tuple2<String, DataPoints>("mst", new
				// DataPoints());
			}
			// else {
			// tuple = new Tuple2<String, DataPoints>("clusters",
			// new DataPoints());
			// }
		}

		return Arrays.asList(mstResult.toString()).iterator();
	}

	public double[] calculateCoreDistances(double[][] dataset, int k,
			DistanceCalculator distanceFunction) {
		int numNeighbors = k - 1;
		double[] coreDistances = new double[dataset.length];

		if (k == 1) {
			for (int point = 0; point < dataset.length; point++) {
				coreDistances[point] = 0;
			}
			return coreDistances;
		}

		for (int point = 0; point < dataset.length; point++) {
			double[] kNNDistances = new double[numNeighbors]; // Sorted nearest
																// distances
																// found so far
			for (int i = 0; i < numNeighbors; i++) {
				kNNDistances[i] = Double.MAX_VALUE;
			}

			for (int neighbor = 0; neighbor < dataset.length; neighbor++) {
				if (point == neighbor)
					continue;
				double distance = distanceFunction
						.computeDistance(dataset[point], dataset[neighbor]);

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
					for (int shiftIndex = numNeighbors
							- 1; shiftIndex > neighborIndex; shiftIndex--) {
						kNNDistances[shiftIndex] = kNNDistances[shiftIndex - 1];
					}
					kNNDistances[neighborIndex] = distance;
				}
			}
			coreDistances[point] = kNNDistances[numNeighbors - 1];
		}
		return coreDistances;
	}

	public MinimumSpanningTree constructMST(double[][] dataset, int[] indices,
			double[] coreDistances, boolean selfEdges,
			DistanceCalculator distanceFunction, int node) {

		int selfEdgeCapacity = 0;
		if (selfEdges)
			selfEdgeCapacity = dataset.length;

		// One bit is set (true) for each attached point, or unset (false) for
		// unattached points:
		BitSet attachedPoints = new BitSet(dataset.length);

		// Each point has a current neighbor point in the tree, and a current
		// nearest distance:
		int[] nearestMRDNeighbors = new int[dataset.length - 1
				+ selfEdgeCapacity];
		int[] nearestneighborsID = new int[dataset.length - 1
				+ selfEdgeCapacity];
		double[] nearestMRDDistances = new double[dataset.length - 1
				+ selfEdgeCapacity];

		for (int i = 0; i < dataset.length - 1; i++) {
			nearestMRDDistances[i] = Double.MAX_VALUE;
		}

		// The MST is expanded starting with the last point in the data set:
		int currentPoint = dataset.length - 1;
		int numAttachedPoints = 1;
		attachedPoints.set(dataset.length - 1);

		// Continue attaching points to the MST until all points are attached:
		while (numAttachedPoints < dataset.length) {
			int nearestMRDPoint = -1;
			double nearestMRDDistance = Double.MAX_VALUE;

			// Iterate through all unattached points, updating distances using
			// the current point:
			for (int neighbor = 0; neighbor < dataset.length; neighbor++) {
				if (currentPoint == neighbor)
					continue;
				if (attachedPoints.get(neighbor) == true)
					continue;

				double distance = distanceFunction.computeDistance(
						dataset[currentPoint], dataset[neighbor]);

				double mutualReachabiltiyDistance = distance;
				if (coreDistances[currentPoint] > mutualReachabiltiyDistance)
					mutualReachabiltiyDistance = coreDistances[currentPoint];
				if (coreDistances[neighbor] > mutualReachabiltiyDistance)
					mutualReachabiltiyDistance = coreDistances[neighbor];

				if (mutualReachabiltiyDistance < nearestMRDDistances[neighbor]) {
					nearestMRDDistances[neighbor] = mutualReachabiltiyDistance;
					nearestMRDNeighbors[neighbor] = indices[currentPoint];
					nearestneighborsID[neighbor] = currentPoint;
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
		int[] otherVertexIndices = new int[dataset.length - 1
				+ selfEdgeCapacity];
		int[] otherVertexIndicesID = new int[dataset.length - 1
				+ selfEdgeCapacity];
		for (int i = 0; i < dataset.length - 1; i++) {
			otherVertexIndices[i] = indices[i];
			otherVertexIndicesID[i] = i;
		}

		// If necessary, attach self edges:
		if (selfEdges) {
			for (int i = dataset.length - 1; i < dataset.length * 2 - 1; i++) {
				int vertex = i - (dataset.length - 1);
				nearestMRDNeighbors[i] = indices[vertex];
				otherVertexIndices[i] = indices[vertex];
				nearestMRDDistances[i] = coreDistances[vertex];
				nearestneighborsID[i] = vertex;
				otherVertexIndicesID[i] = vertex;
			}
		}

		LinkedList<MinimumSpanningTree> mst = new LinkedList<MinimumSpanningTree>();
		for (int i = 0; i < nearestMRDNeighbors.length; i++) {
			mst.add(new MinimumSpanningTree(nearestMRDNeighbors[i],
					otherVertexIndices[i], nearestMRDDistances[i],
					nearestneighborsID[i], otherVertexIndicesID[i], node));
		}

		return new MinimumSpanningTree(mst);

		// return new UndirectedGraph(dataset.size(), nearestMRDNeighbors,
		// otherVertexIndices, nearestMRDDistances);
	}

	public Integer getK() {
		return k;
	}

	public void setK(Integer k) {
		this.k = k;
	}

	public void setProcessU(Integer pro) {
		this.processU = pro;
	}

	public Integer getProcessU() {
		return processU;
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

	public String getInputFile() {
		return inputFile;
	}

	public void setInputFile(String inputFile) {
		this.inputFile = inputFile;
	}

	public int getCountLevels() {
		return countLevels;
	}

	public void setCountLevels(int countLevels) {
		this.countLevels = countLevels;
	}
}
