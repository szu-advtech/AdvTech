package databubbles;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.TreeSet;

import distance.DistanceCalculator;
import hdbscanstar.Cluster;
import hdbscanstar.Clusters;
import hdbscanstar.OutlierScore;
import hdbscanstar.UndirectedGraph;

public class HdbscanDataBubbles implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private LinkedList<Clusters> clusters;
	private int[] vertice1;

	private int[] vertice2;
	private double[] dmreach;

	public HdbscanDataBubbles() {
		this.clusters = null;
		this.vertice1 = null;
		this.vertice2 = null;
		this.dmreach = null;
	}

	// ------------------------------ PRIVATE VARIABLES
	// ------------------------------
	// ------------------------------ CONSTANTS ------------------------------
	public static final String WARNING_MESSAGE = "----------------------------------------------- WARNING -----------------------------------------------\n"
			+ "With your current settings, the K-NN density estimate is discontinuous as it is not well-defined\n"
			+ "(infinite) for some data objects, either due to replicates in the data (not a set) or due to numerical\n"
			+ "roundings. This does not affect the construction of the density-based clustering hierarchy, but\n"
			+ "it affects the computation of cluster stability by means of relative excess of mass. For this reason,\n"
			+ "the post-processing routine to extract a flat partition containing the most stable clusters may\n"
			+ "produce unexpected results. It may be advisable to increase the value of MinPts and/or M_clSize.\n"
			+ "-------------------------------------------------------------------------------------------------------";

	private static final int FILE_BUFFER_SIZE = 32678;

	// ------------------------------ CONSTRUCTORS
	// ------------------------------
	// ------------------------------ PUBLIC METHODS
	// ------------------------------

	/**
	 * Calculates the core distances for each point in the data set, given some
	 * value for k.
	 * 
	 * @param dataSet
	 *            A double[][] where index [i][j] indicates the jth attribute of
	 *            data point i
	 * @param k
	 *            Each point's core distance will be it's distance to the kth
	 *            nearest neighbor
	 * @param distanceFunction
	 *            A DistanceCalculator to compute distances between points
	 * @return An array of core distances
	 */
	public double[] calculateCoreDistancesBubbles(double[][] repB, int[] nB, double[] eB, double[] nnDistB, int k,
			DistanceCalculator distanceFunction) {
		int numNeighbors = k - 1; /* MinPoints */

		int[] indexBubbles = new int[numNeighbors];

		for (int i = 0; i < indexBubbles.length; i++) {
			indexBubbles[i] = 0;
		}

		double[] coreDistances = new double[repB.length];

		if (k == 1) {
			return coreDistances;
		}

		for (int point = 0; point < repB.length; point++) {
			double[] kNNDistances = new double[numNeighbors]; // Sorted nearest distances found so far
			for (int i = 0; i < numNeighbors; i++) {
				kNNDistances[i] = Double.MAX_VALUE;
			}

			for (int neighbor = 0; neighbor < repB.length; neighbor++) {
				if (point == neighbor)
					continue;
				// distance of the Data Bubbles
				double distance = distanceFunction.computeDistance(repB[point], repB[neighbor]);
				distance = distanceBubbles(distance, eB, nnDistB, point, neighbor);

				// Check at which position in the nearest distances the current
				// distance would fit:
				int neighborIndex = numNeighbors;
				while (neighborIndex >= 1 && distance < kNNDistances[neighborIndex - 1]) {
					neighborIndex--;
				}

				// Shift elements in the array to make room for the current
				// distance:
				if (neighborIndex < numNeighbors) {
					for (int shiftIndex = numNeighbors - 1; shiftIndex > neighborIndex; shiftIndex--) {
						kNNDistances[shiftIndex] = kNNDistances[shiftIndex - 1];
					}
					kNNDistances[neighborIndex] = distance;
					indexBubbles[neighborIndex] = neighbor;
				}
			}
			if (nB[point] >= numNeighbors) {
				coreDistances[point] = Math.pow((numNeighbors / nB[point]), (1 / repB[point].length)) * eB[point];
			} else {
				int nX = nB[point];
				int i = 0;
				int index = 0;
				while (nX < numNeighbors) {
					nX += nB[indexBubbles[i]];
					index = indexBubbles[i];
					i += 1;
				}
				int sum = nB[point];
				int aux = 0;
				for (int j = 0; j < i; j++) {
					double distanceC = distanceFunction.computeDistance(repB[indexBubbles[j]], repB[i]);
					distanceC = distanceBubbles(distanceC, eB, nnDistB, indexBubbles[j], i);
					if (sum < numNeighbors && kNNDistances[j] < distanceC) {
						aux = numNeighbors - sum;
					}
					sum += nB[indexBubbles[j]];
				}
				coreDistances[point] = kNNDistances[i] + Math.pow((aux / nB[i]), 1 / repB[i].length) * eB[i];
			}
		}
		return coreDistances;
	}

	/**
	 * Constructs the minimum spanning tree of mutual reachability distances for
	 * the data set, given the core distances for each point.
	 * 
	 * @param dataSet
	 *            A double[][] where index [i][j] indicates the jth attribute of
	 *            data point i
	 * @param coreDistances
	 *            An array of core distances for each data point
	 * @param selfEdges
	 *            If each point should have an edge to itself with weight equal
	 *            to core distance
	 * @param distanceFunction
	 *            A DistanceCalculator to compute distances between points
	 * @return An MST for the data set using the mutual reachability distances
	 * @throws IOException
	 */
	public UndirectedGraph constructMSTBubbles(double[][] dataRepB, int[] nB, double[] eB, double[] nnDistB,
			int[] idBubbles, double[] coreDistances, boolean selfEdges, DistanceCalculator distanceFunction)
			throws IOException {

		int selfEdgeCapacity = 0;
		if (selfEdges) {
			selfEdgeCapacity = dataRepB.length;
		}

		// One bit is set (true) for each attached point, or unset (false) for
		// unattached points:
		BitSet attachedPoints = new BitSet(dataRepB.length);

		// Each point has a current neighbor point in the tree, and a current
		// nearest distance:
		int[] nearestMRDNeighbors = new int[dataRepB.length - 1 + selfEdgeCapacity];
		double[] nearestMRDDistances = new double[dataRepB.length - 1 + selfEdgeCapacity];

		for (int i = 0; i < dataRepB.length - 1; i++) {
			nearestMRDDistances[i] = Double.MAX_VALUE;
		}

		// The MST is expanded starting with the last point in the data set:
		int currentPoint = dataRepB.length - 1;
		int numAttachedPoints = 1;
		attachedPoints.set(dataRepB.length - 1);

		// Continue attaching points to the MST until all points are attached:
		while (numAttachedPoints < dataRepB.length) {
			int nearestMRDPoint = -1;
			double nearestMRDDistance = Double.MAX_VALUE;

			// Iterate through all unattached points, updating distances using
			// the current point:

			for (int neighbor = 0; neighbor < dataRepB.length; neighbor++) {
				if (currentPoint == neighbor) {
					continue;
				}
				if (attachedPoints.get(neighbor) == true) {
					continue;
				}

				double distance = distanceFunction.computeDistance(dataRepB[currentPoint], dataRepB[neighbor]);
				distance = distanceBubbles(distance, eB, nnDistB, currentPoint, neighbor);

				double mutualReachabilityDistance = distance;
				if (coreDistances[currentPoint] > mutualReachabilityDistance) {
					mutualReachabilityDistance = coreDistances[currentPoint];
				}
				if (coreDistances[neighbor] > mutualReachabilityDistance) {
					mutualReachabilityDistance = coreDistances[neighbor];
				}

				if (mutualReachabilityDistance < nearestMRDDistances[neighbor]) {
					nearestMRDDistances[neighbor] = mutualReachabilityDistance;
					nearestMRDNeighbors[neighbor] = idBubbles[currentPoint];
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
		int[] otherVertexIndices = new int[dataRepB.length - 1 + selfEdgeCapacity];
		for (int i = 0; i < dataRepB.length - 1; i++) {
			otherVertexIndices[i] = idBubbles[i]; // // indices[i]
		}

		// If necessary, attach self edges:
		if (selfEdges) {
			for (int i = dataRepB.length - 1; i < dataRepB.length * 2 - 1; i++) {
				int vertex = i - (dataRepB.length - 1);
				nearestMRDNeighbors[i] = idBubbles[vertex]; // indices[vertex]
				otherVertexIndices[i] = idBubbles[vertex]; // indices[vertex]
				nearestMRDDistances[i] = coreDistances[vertex]; // indices[vertex]
			}
		}
		return new UndirectedGraph(dataRepB.length, nearestMRDNeighbors, otherVertexIndices, nearestMRDDistances);
	}

	public void constructClusterTree(UndirectedGraph mst, int mclSize, int[] nB) {
		// The current edge being removed from the MST:
		int currentEdgeIndex = mst.getNumEdges() - 1;
		int nextClusterLabel = 2;

		// The previous and current cluster numbers of each point in the data
		// set:
		int[] currentClusterLabels = new int[mst.getNumVertices()];
		for (int i = 0; i < currentClusterLabels.length; i++) {
			currentClusterLabels[i] = 1;
		}

		// A list of clusters in the simplified cluster tree
		this.clusters = new LinkedList<Clusters>();
		// int label, int parent, double birthLevel, int numPoints, int[]
		// members
		int allMembers = 0;
		for (int i = 0; i < nB.length; i++) {
			allMembers += nB[i];
		}
		clusters.add(new Clusters(1, -1, Double.NaN, allMembers, null));
		// Sets for the clusters and vertices that are affected by the edge(s)
		// being removed:
		while (currentEdgeIndex >= 0) {
			Map<Integer, TreeSet<Integer>> affectedVerticesByLabel = new HashMap<Integer, TreeSet<Integer>>();
			double currentEdgeWeight = mst.getEdgeWeightAtIndex(currentEdgeIndex);
			// Remove all edges tied with the current edge weight, and store
			// relevant clusters and vertices:
			while (currentEdgeIndex >= 0 && mst.getEdgeWeightAtIndex(currentEdgeIndex) == currentEdgeWeight) {
				int firstVertex = mst.getFirstVertexAtIndex(currentEdgeIndex);
				int secondVertex = mst.getSecondVertexAtIndex(currentEdgeIndex);
				mst.getEdgeListForVertex(firstVertex).remove((Integer) secondVertex);
				mst.getEdgeListForVertex(secondVertex).remove((Integer) firstVertex);

				if (currentClusterLabels[firstVertex] == 0) {
					currentEdgeIndex--;
					continue;
				}
				if (affectedVerticesByLabel.get(currentClusterLabels[firstVertex]) == null) {
					affectedVerticesByLabel.put(currentClusterLabels[firstVertex], new TreeSet<Integer>());
				}
				affectedVerticesByLabel.get(currentClusterLabels[firstVertex]).add(firstVertex);
				affectedVerticesByLabel.get(currentClusterLabels[firstVertex]).add(secondVertex);
				currentEdgeIndex--;
			}
			if (affectedVerticesByLabel.isEmpty()) {
				continue;
			}

			for (Integer parentLabel : affectedVerticesByLabel.keySet()) {
				ArrayList<Clusters> newClusters = new ArrayList<Clusters>();
				while (!affectedVerticesByLabel.get(parentLabel).isEmpty()) {
					int rootVertex = affectedVerticesByLabel.get(parentLabel).pollFirst();
					boolean[] visited = new boolean[mst.getNumVertices()];
					for (int i = 0; i < visited.length; i++) {
						visited[i] = false;
					}
					TreeSet<Integer> queue = new TreeSet<Integer>();
					TreeSet<Integer> subComponent = new TreeSet<Integer>();
					visited[rootVertex] = true;
					queue.add(rootVertex);
					subComponent.add(rootVertex);
					while (!queue.isEmpty()) {
						int vertex = queue.pollFirst();
						subComponent.add(vertex);
						// for each adjacent vertex of neighbor
						for (int adj = 0; adj < mst.getEdges().get(vertex).size(); adj++) {
							if (!visited[mst.getEdges().get(vertex).get(adj)]) {
								queue.add(mst.getEdges().get(vertex).get(adj));
								visited[mst.getEdges().get(vertex).get(adj)] = true;
								subComponent.add(mst.getEdges().get(vertex).get(adj)); // teste
							}
						}
					}
					int countMembers = 0;
					Iterator<Integer> iter = subComponent.iterator();
					while (iter.hasNext()) {
						countMembers += nB[iter.next()];
					}
					if (countMembers >= mclSize) {
						newClusters.add(
								new Clusters(parentLabel, parentLabel, currentEdgeWeight, countMembers, subComponent));
					} else {
						iter = subComponent.iterator();
						while (iter.hasNext()) {
							currentClusterLabels[iter.next()] = 0;
						}
						for (int i = 0; i < clusters.size(); i++) {
							if ((clusters.get(i).getLabel() == parentLabel)
									&& (clusters.get(i).getDeathLevel() == Double.MAX_VALUE)) {
								clusters.get(i).detachPoints(countMembers, 0, currentEdgeWeight);
								break;
							}
						}
					}
				}
				if (newClusters.size() >= 2) {
					// int count = 0;
					while (!newClusters.isEmpty()) {
						Clusters cluster = newClusters.remove(0);
						cluster.setLabel(nextClusterLabel);
						Iterator<Integer> iter = cluster.getMembers().iterator();
						while (iter.hasNext()) {
							currentClusterLabels[iter.next()] = nextClusterLabel;
						}
						nextClusterLabel++;
						for (int i = 0; i < clusters.size(); i++) {
							if ((clusters.get(i).getLabel() == cluster.getParentId())
									&& (clusters.get(i).getDeathLevel() == Double.MAX_VALUE)) {
								clusters.get(i).setHasChildren(true);// ***
								clusters.get(i).detachPoints(cluster.getNumPoints(), 0, cluster.getBirthLevel());
								break;
							}
						}
						clusters.add(cluster);
					}
				}
			}
		}
	}

	public int[][] findProminentClustersAndClassificationNoiseBubbles(LinkedList<Clusters> clusterTree, double[][] rep,
			int[] nB, double[] extent, double[] nnDist, DistanceCalculator distanceFunction, int[] idBubbles) {
		clusterTree.remove(0); // remove root node
		Comparator<Clusters> comparatorA = new Comparator<Clusters>() {
			public int compare(Clusters o1, Clusters o2) {
				if (o1.getBirthLevel() > o2.getBirthLevel()) {
					return 1;
				}
				if (o1.getBirthLevel() < o2.getBirthLevel()) {
					return -1;
				}
				return 0;
			}
		};
		Map<Integer, LinkedList<double[]>> adjListNodes = new HashMap<Integer, LinkedList<double[]>>();
		for (Clusters parent : clusterTree) {
			if (!parent.isHasChildren()) {
				if (adjListNodes.get(parent.getLabel()) == null) {
					adjListNodes.put(parent.getLabel(), new LinkedList<double[]>());
				}
			}
			for (Clusters child : clusterTree) {
				if (parent.getLabel() == child.getParentId()) {
					if (adjListNodes.get(parent.getLabel()) == null) {
						adjListNodes.put(parent.getLabel(), new LinkedList<double[]>());
					}
					double[] v = new double[5];
					v[0] = parent.getStability();
					v[1] = child.getLabel();
					v[2] = child.getStability();
					v[3] = 1.0;
					v[4] = parent.getParentId();
					adjListNodes.get(parent.getLabel()).add(v);
				}
			}
		}
		// sorting from birthlevel
		Collections.sort(clusterTree, comparatorA);
		int[][] flat = new int[nB.length][2];
		for (int object = 0; object < nB.length; object++) {
			flat[object][0] = idBubbles[object];
			flat[object][1] = 0;
		}
		// finding the most significant clusters at the clusters tree;
		TreeSet<Integer> solution = new TreeSet<Integer>();
		LinkedList<Integer> setKeys = new LinkedList<Integer>();
		for (Clusters cl : clusterTree) {
			solution.add(cl.getLabel());
			setKeys.add(cl.getLabel());
		}
		for (Integer key : setKeys) {
			double childrenStability = 0.0;
			if (!adjListNodes.get(key).isEmpty()) {
				for (double[] childrenInfo : adjListNodes.get(key)) {
					childrenStability += childrenInfo[2];
				}
				if (childrenStability <= adjListNodes.get(key).get(0)[0]) {
					for (int i = 0; i < adjListNodes.get(key).size(); i++) {
						TreeSet<Integer> queue = new TreeSet<Integer>();
						TreeSet<Integer> visited = new TreeSet<Integer>();
						int rootVertex = (int) adjListNodes.get(key).get(i)[1];

						visited.add(rootVertex);
						queue.add(rootVertex);
						adjListNodes.get(key).get(i)[3] = 0.0;
						// remove children nodes from the overall solution;
						solution.remove(rootVertex);
						while (!queue.isEmpty()) {
							int vertex = queue.pollFirst();
							// for each adjacent vertex of neighbor
							if (adjListNodes.get(vertex) != null) {
								for (int adj = 0; adj < adjListNodes.get(vertex).size(); adj++) {
									solution.remove(vertex);
									if (!visited.contains((int) adjListNodes.get(vertex).get(adj)[1])) {
										queue.add((int) adjListNodes.get(vertex).get(adj)[1]);
										visited.add((int) adjListNodes.get(vertex).get(adj)[1]);
									}
								}
							}
						}
					}
				} else {
					adjListNodes.get(key).get(0)[0] = childrenStability;
					if (adjListNodes.get((int) adjListNodes.get(key).get(0)[4]) != null)
						for (int adj = 0; adj < adjListNodes.get((int) adjListNodes.get(key).get(0)[4]).size(); adj++) {
							if ((int) adjListNodes.get((int) adjListNodes.get(key).get(0)[4]).get(adj)[1] == key
									.intValue()) {
								adjListNodes.get((int) adjListNodes.get(key).get(0)[4]).get(adj)[2] = childrenStability;
							}
						}
				}
			} else {
				solution.remove(key);
			}
		}
		for (Clusters cl : clusterTree) {
			for (Integer prominent : solution) {
				if (cl.getLabel() == prominent) {
					for (Integer member : cl.getMembers()) {
						flat[member][1] = prominent;
					}
				}
			}
		}

		// compute the distance between noise bubbles and non-noise bubbles and
		// include the noise
		// to its nearest valid cluster (of bubbles).
		for (int point = 0; point < nB.length; point++) {
			double minDistance = Double.MAX_VALUE;
			int nearestNeighbor = 0;
			for (int neighbor = 0; neighbor < nB.length; neighbor++) {
				if (point == neighbor) {
					continue;
				}
				if ((flat[point][1] == 0) && (flat[neighbor][1] != 0)) {
					double distance = distanceFunction.computeDistance(rep[point], rep[neighbor]);
					distance = distanceBubbles(distance, extent, nnDist, point, neighbor);
					if (distance < minDistance) {
						minDistance = distance;
						nearestNeighbor = neighbor;
						flat[point][1] = flat[neighbor][1];
					}
				}
			}
		}
		return flat;
	}

	public void findInterClusterEdges(UndirectedGraph mst, int[][] flat) {
		ArrayList<Integer> vertice1 = new ArrayList<Integer>();
		ArrayList<Integer> vertice2 = new ArrayList<Integer>();
		ArrayList<Double> dmreach = new ArrayList<Double>();
		for (int i = 0; i < mst.getNumEdges(); i++) {
			if (flat[mst.getFirstVertexAtIndex(i)][1] != flat[mst.getSecondVertexAtIndex(i)][1]) {
				vertice1.add(mst.getFirstVertexAtIndex(i));
				vertice2.add(mst.getSecondVertexAtIndex(i));
				dmreach.add(mst.getEdgeWeightAtIndex(i));
			}
		}
		if (vertice1.size() > 0) {
			this.vertice1 = new int[vertice1.size()];
			this.vertice2 = new int[vertice2.size()];
			this.dmreach = new double[dmreach.size()];
			for (int i = 0; i < this.vertice1.length; i++) {
				this.vertice1[i] = vertice1.get(i);
				this.vertice2[i] = vertice2.get(i);
				this.dmreach[i] = dmreach.get(i);
			}
		}
	}

	/**
	 * Produces the outlier score for each point in the data set, and returns a
	 * sorted list of outlier scores. propagateTree() must be called before
	 * calling this method.
	 * 
	 * @param clusters
	 *            A list of Clusters forming a cluster tree which has already
	 *            been propagated
	 * @param pointNoiseLevels
	 *            A double[] with the levels at which each point became noise
	 * @param pointLastClusters
	 *            An int[] with the last label each point had before becoming
	 *            noise
	 * @param coreDistances
	 *            An array of core distances for each data point
	 * @param outlierScoresOutputFile
	 *            The path to the outlier scores output file
	 * @param delimiter
	 *            The delimiter for the output file
	 * @param infiniteStability
	 *            true if there are any clusters with infinite stability, false
	 *            otherwise
	 * @return An ArrayList of OutlierScores, sorted in descending order
	 * @throws IOException
	 *             If any errors occur opening or writing to the output file
	 */
	public static ArrayList<OutlierScore> calculateOutlierScoresBubbles(ArrayList<Cluster> clusters,
			double[] pointNoiseLevels, int[] pointLastClusters, double[] coreDistances, String outlierScoresOutputFile,
			String delimiter, boolean infiniteStability, Integer iter) throws IOException {

		int numPoints = pointNoiseLevels.length;
		ArrayList<OutlierScore> outlierScores = new ArrayList<OutlierScore>(numPoints);

		// Iterate through each point, calculating its outlier score:
		for (int i = 0; i < numPoints; i++) {
			double epsilon_max = clusters.get(pointLastClusters[i]).getPropagatedLowestChildDeathLevel();
			double epsilon = pointNoiseLevels[i];

			double score = 0;
			if (epsilon != 0) {
				score = 1 - (epsilon_max / epsilon);
			}

			outlierScores.add(new OutlierScore(score, coreDistances[i], i));
		}

		// Sort the outlier scores:
		Collections.sort(outlierScores);

		// Output the outlier scores:
		BufferedWriter writer = new BufferedWriter(new FileWriter(outlierScoresOutputFile), FILE_BUFFER_SIZE);
		if (infiniteStability) {
			writer.write(WARNING_MESSAGE + "\n");
		}

		for (OutlierScore outlierScore : outlierScores) {
			writer.write(outlierScore.getScore() + delimiter + outlierScore.getId() + "\n");
		}
		writer.close();

		return outlierScores;
	}

	public static double distanceBubbles(double distance, double[] eB, double[] nnDistB, int point, int neighbor) {
		double verify = distance - (eB[point] + eB[neighbor]);
		if (verify >= 0) {
			distance = (distance - (eB[point] + eB[neighbor])) + (nnDistB[point] + nnDistB[neighbor]);
		} else {
			distance = Math.max(nnDistB[point], nnDistB[neighbor]);
		}
		return distance;
	}
	// ------------------------------ GETTERS & SETTERS
	// ------------------------------

	public LinkedList<Clusters> getClusters() {
		return clusters;
	}

	public void setClusters(LinkedList<Clusters> clusters) {
		this.clusters = clusters;
	}

	public int[] getVertice1() {
		return vertice1;
	}

	public void setVertice1(int[] vertice1) {
		this.vertice1 = vertice1;
	}

	public int[] getVertice2() {
		return vertice2;
	}

	public void setVertice2(int[] vertice2) {
		this.vertice2 = vertice2;
	}

	public double[] getDmreach() {
		return dmreach;
	}

	public void setDmreach(double[] dmreach) {
		this.dmreach = dmreach;
	}

}
