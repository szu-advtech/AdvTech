/*package partition.mappers;

import hdbscanstar.Cluster;
import hdbscanstar.Constraint;
import hdbscanstar.OutlierScore;
import hdbscanstar.UndirectedGraph;
import hdbscanstar.Constraint.CONSTRAINT_TYPE;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import main.Main;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;
import databubbles.HdbscanDataBubbles;
import datastructure.ClusterFeatureDataBubbles;
import datastructure.DataBubbles;
import datastructure.DataPoints;
import distance.DistanceCalculator;

public class HDBSCANSTARMapper implements
		PairFlatMapFunction<Iterator<Tuple2<String, ClusterFeatureDataBubbles>>, String, ClusterFeatureDataBubbles> {

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

	private static final long serialVersionUID = 1L;
	private Integer processingU;
	// private LinkedList<LinkedList<DataPoints>> listOfList;
	private DistanceCalculator distanceFunction;
	private Integer minPoints;
	private Integer minClusterSize;
	private String inputFile;
	private String constraintsFile;
	private Integer k;
	private boolean compactHierarchy;
	private Integer processing_units;

	private String hierarchyFile;
	private String interEdgesFile;
	private String clusterTreeFile;
	private String partitionFile;
	private String outlierScoreFile;
	private String visualizationFile;
	private int countLevels;

	public HDBSCANSTARMapper(Integer processing_units,
			DistanceCalculator distanceFunction, Integer minPoints,
			Integer minClusterSize, String hierarchyFile,
			String clusterTreeFile, boolean compactHierarchy,
			String constraintsFile, String interEdgesFile,
			String visualizationFile, String inputFile, String outlierScoreFile,
			String partitionFile, Integer k, int countLevels) {

		this.setProcessingU(processing_units);
		this.setDistanceFunction(distanceFunction);
		this.setMinPoints(minPoints);
		this.setMinClusterSize(minClusterSize);
		this.setHierarchyFile(hierarchyFile);
		this.setClusterTreeFile(clusterTreeFile);
		this.setCompactHierarchy(compactHierarchy);
		this.setConstraintsFile(constraintsFile);
		this.setInterEdgesFile(interEdgesFile);
		this.setVisualizationFile(visualizationFile);
		this.setInputFile(inputFile);
		this.setOutlierScoreFile(outlierScoreFile);
		this.setPartitionFile(partitionFile);
		this.setK(k);
		this.setCountLevels(countLevels);
	}

	public Iterator<Tuple2<String, ClusterFeatureDataBubbles>> call(
			Iterator<Tuple2<String, ClusterFeatureDataBubbles>> t)
			throws Exception {
		LinkedList<ClusterFeatureDataBubbles> cfs = new LinkedList<ClusterFeatureDataBubbles>();
		while (t.hasNext()) {
			cfs.add(t.next()._2());
		}

		if (cfs.isEmpty()) {
			return new LinkedList<Tuple2<String, ClusterFeatureDataBubbles>>().iterator();
		}

		double[][] rep = new double[cfs.size()][cfs.get(0).getRep().length];
		double[] extent = new double[cfs.size()];
		double[] nnDist = new double[cfs.size()];
		int[] n = new int[cfs.size()];
		String[] keysBubbles = new String[cfs.size()];

		for (int i = 0; i < rep.length; i++) {
			for (int j = 0; j < rep[i].length; j++) {
				rep[i][j] = cfs.get(i).getRep()[j];
			}
			extent[i] = cfs.get(i).getExtent();
			n[i] = cfs.get(i).getN();
			nnDist[i] = cfs.get(i).getNnDist();
			keysBubbles[i] = cfs.get(i).getKey();
		}

		String[] auxNode = keysBubbles[0].split("\\.");

		Configuration configuration = new Configuration();
		configuration.addResource(
				new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
		configuration.addResource(
				new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));
		FileSystem fs = FileSystem.get(configuration);

		//
		// DataBubbles bubbles = new DataBubbles();
		double[] coreDistances = HdbscanDataBubbles
				.calculateCoreDistancesBubbles(rep, n, extent, nnDist,
						this.minPoints, this.distanceFunction);
		UndirectedGraph mst = HdbscanDataBubbles.constructMSTBubbles(rep, n,
				extent, nnDist, coreDistances, true, this.distanceFunction);
		mst.quicksortByEdgeWeight();

		// Construir hierarquia sobre os Data Bubbles
		int[] idB = new int[rep.length];
		String[] keys = new String[rep.length];
		int[] id_cluster = new int[rep.length];

		for (int i = 0; i < idB.length; i++) {
			String[] tmpKey = keysBubbles[i].split("\\.");
			idB[i] = Integer.parseInt(tmpKey[1]);
			keys[i] = keysBubbles[i];
		}

		// Remove references to unneeded objects:

		double[] pointNoiseLevels = new double[rep.length];
		int[] pointLastClusters = new int[rep.length];

		// Compute hierarchy and cluster tree:
		ArrayList<Cluster> clusters = null;
		clusters = computeHierarchyAndClusterTreeBubbles(mst,
				this.minClusterSize, this.compactHierarchy, null,
				this.hierarchyFile, this.clusterTreeFile, ",", pointNoiseLevels,
				pointLastClusters, this.visualizationFile, this.inputFile, n,
				auxNode[0], configuration, fs);

		// Remove references to unneeded objects:

		// Propagate clusters:
		boolean infiniteStability = HdbscanDataBubbles
				.propagateTreeBubbles(clusters);

		// Compute final flat partitioning:
		id_cluster = findProminentClustersBubbles(clusters, this.hierarchyFile,
				this.partitionFile, ",", rep.length, infiniteStability,
				auxNode[0], configuration, fs, this.inputFile);

		for (int i = 0; i < id_cluster.length; i++) {
			if (id_cluster[i] == 0) {
				double minDistance = Double.MAX_VALUE;
				int minIndex = 0;
				for (int j = 0; j < id_cluster.length; j++) {
					if (i == j)
						continue;
					if (id_cluster[j] != 0) {
						// calculate distance;
						double distance = this.distanceFunction
								.computeDistance(rep[i], rep[j]);
						double distBubble = HdbscanDataBubbles.distanceBubbles(
								distance, extent, nnDist, i, j);
						if (distBubble < minIndex) {
							minDistance = distBubble;
							minIndex = j;
						}
					}
				}
				id_cluster[i] = minIndex;
			}
		}
		
		// mapear bubbles e reduzir
		Set<Integer> setTree = new TreeSet<Integer>();
		for (int i = 0; i < id_cluster.length; i++) {
			setTree.add(id_cluster[i]);
		}
		int[] indexAux = new int[setTree.size()];
		int[] indexAux2 = new int[setTree.size()];
		int[] nodeAux = new int[setTree.size()];
		int[] tamClusters = new int[setTree.size()];
		Iterator<Integer> iterator = setTree.iterator();
		int a = 0;
		while (iterator.hasNext()) {
			indexAux[a] = iterator.next();
			indexAux2[a] = a;
			nodeAux[a] = ++Main.nodeCount;
			a++;
		}

		int[] nodesID = new int[id_cluster.length];

		for (int i = 0; i < indexAux.length; i++) {
			for (int j = 0; j < id_cluster.length; j++) {
				if (id_cluster[j] == indexAux[i]) {
					id_cluster[j] = indexAux2[i];
					nodesID[j] = nodeAux[i];
				}
			}
		}

		int max = 0;
		for (int i = 0; i < setTree.size(); i++) {
			if (indexAux2[i] > max) {
				max = indexAux2[i];
			}
		}

		for (int i = 0; i < id_cluster.length; i++) {
			tamClusters[id_cluster[i]] += n[i];
		}

		int[] tam = new int[id_cluster.length];
		for (int i = 0; i < tamClusters.length; i++) {
			for (int j = 0; j < id_cluster.length; j++) {
				if (id_cluster[j] == i) {
					tam[j] = tamClusters[i];
				}
			}
		}
		int numOfClusters = setTree.size();

		Path pathMSTBubbles = new Path(
				this.inputFile + "/resultados/mstBubbles/" + countLevels + "_"
						+ auxNode[0] + "/part-00000");
		BufferedWriter mstWriter;
		mstWriter = new BufferedWriter(
				new OutputStreamWriter(fs.create(pathMSTBubbles, true)));

		for (int i = 0; i < mst.getNumEdges(); i++) {
			mstWriter.write(mst.getFirstVertexAtIndex(i) + " "
					+ mst.getSecondVertexAtIndex(i) + " "
					+ mst.getEdgeWeightAtIndex(i) + "\n");
		}
		mstWriter.close();

		for (int i = 0; i < id_cluster.length; i++) {
			if (id_cluster[i] == 0) {
				double minDistance = Double.MAX_VALUE;
				int minIndex = 0;
				for (int j = 0; j < id_cluster.length; j++) {
					if (id_cluster[j] != 0) {
						// calculate distance;
						double distance = this.distanceFunction
								.computeDistance(rep[i], rep[j]);
						double distBubble = HdbscanDataBubbles.distanceBubbles(
								distance, extent, nnDist, i, j);
						if (distBubble < minIndex) {
							minDistance = distBubble;
							minIndex = j;
						}
					}
				}
				id_cluster[i] = minIndex;
			}
		}

		DataBubbles dataB = new DataBubbles(keys, nodesID, id_cluster, tam, idB,
				numOfClusters, coreDistances, rep, extent, nnDist, n);
		// v1._2.setBubbles(dataB);
		writeDataBubbles(dataB, this.inputFile, auxNode[0]);

		LinkedList<Tuple2<String, ClusterFeatureDataBubbles>> l = new LinkedList<Tuple2<String, ClusterFeatureDataBubbles>>();

		l.add(new Tuple2<String, ClusterFeatureDataBubbles>(auxNode[0],
				new ClusterFeatureDataBubbles()));

		return l.iterator();
	}

	// /////////////////////////////////////////////////////////////////////////////////////////
	public void writeDataBubbles(DataBubbles bubbles, String inputFile,
			String node) throws IOException {
		Configuration configuration = new Configuration();
		configuration.addResource(
				new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
		configuration.addResource(
				new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));
		FileSystem fs = FileSystem.get(configuration);
		Path pathDataBubbles = new Path(inputFile + "/resultados/"
				+ "dataBubbles/" + countLevels + "/b_" + node);
		OutputStream out = fs.create(pathDataBubbles);
		ObjectOutputStream objOut = new ObjectOutputStream(out);
		objOut.writeObject(bubbles);
		objOut.close();
		// out.close();
	}

	public ArrayList<Cluster> computeHierarchyAndClusterTreeBubbles(
			UndirectedGraph mst, int minClusterSize, boolean compactHierarchy,
			ArrayList<Constraint> constraints, String hierarchyOutputFile,
			String treeOutputFile, String delimiter, double[] pointNoiseLevels,
			int[] pointLastClusters, String visualizationOutputFile,
			String inputFile, int[] n, String node, Configuration configuration,
			FileSystem fs) throws IOException {

		Path pathHierarchy = new Path(inputFile + "/resultados/hdbscan/"
				+ countLevels + "/" + node + "/" + hierarchyOutputFile);
		BufferedWriter hierarchyWriter;

		hierarchyWriter = new BufferedWriter(
				new OutputStreamWriter(fs.create(pathHierarchy, true)));

		long hierarchyCharsWritten = 0;

		int lineCount = 0; // Indicates the number of lines written into
		// hierarchyFile.

		// The current edge being removed from the MST:
		int currentEdgeIndex = mst.getNumEdges() - 1;

		int nextClusterLabel = 2;
		boolean nextLevelSignificant = true;

		// The previous and current cluster numbers of each point in the data
		// set:
		int[] previousClusterLabels = new int[mst.getNumVertices()];
		int[] currentClusterLabels = new int[mst.getNumVertices()];
		for (int i = 0; i < currentClusterLabels.length; i++) {
			currentClusterLabels[i] = 1;
			previousClusterLabels[i] = 1;
		}

		// A list of clusters in the cluster tree, with the 0th cluster (noise)
		// null:
		ArrayList<Cluster> clusters = new ArrayList<Cluster>();
		clusters.add(null);
		clusters.add(new Cluster(1, null, Double.NaN, mst.getNumVertices()));

		// Calculate number of constraints satisfied for cluster 1:
		TreeSet<Integer> clusterOne = new TreeSet<Integer>();
		clusterOne.add(1);
		calculateNumConstraintsSatisfiedBubbles(clusterOne, clusters,
				constraints, currentClusterLabels);

		// Sets for the clusters and vertices that are affected by the edge(s)
		// being removed:
		TreeSet<Integer> affectedClusterLabels = new TreeSet<Integer>();
		TreeSet<Integer> affectedVertices = new TreeSet<Integer>();

		while (currentEdgeIndex >= 0) {
			double currentEdgeWeight = mst
					.getEdgeWeightAtIndex(currentEdgeIndex);
			ArrayList<Cluster> newClusters = new ArrayList<Cluster>();

			// Remove all edges tied with the current edge weight, and store
			// relevant clusters and vertices:
			while (currentEdgeIndex >= 0 && mst.getEdgeWeightAtIndex(
					currentEdgeIndex) == currentEdgeWeight) {
				int firstVertex = mst.getFirstVertexAtIndex(currentEdgeIndex);
				int secondVertex = mst.getSecondVertexAtIndex(currentEdgeIndex);
				mst.getEdgeListForVertex2(firstVertex)
						.remove((Integer) secondVertex);
				mst.getEdgeListForVertex2(secondVertex)
						.remove((Integer) firstVertex);

				if (currentClusterLabels[firstVertex] == 0) {
					currentEdgeIndex--;
					continue;
				}

				affectedVertices.add(firstVertex);
				affectedVertices.add(secondVertex);
				affectedClusterLabels.add(currentClusterLabels[firstVertex]);
				currentEdgeIndex--;
			}

			if (affectedClusterLabels.isEmpty()) {
				continue;
			}

			// Check each cluster affected for a possible split:
			while (!affectedClusterLabels.isEmpty()) {
				int examinedClusterLabel = affectedClusterLabels.last();
				affectedClusterLabels.remove(examinedClusterLabel);
				TreeSet<Integer> examinedVertices = new TreeSet<Integer>();

				// Get all affected vertices that are members of the cluster
				// currently being examined:
				Iterator<Integer> vertexIterator = affectedVertices.iterator();
				while (vertexIterator.hasNext()) {
					int vertex = vertexIterator.next();

					if (currentClusterLabels[vertex] == examinedClusterLabel) {
						examinedVertices.add(vertex);
						vertexIterator.remove();
					}
				}

				TreeSet<Integer> firstChildCluster = null;
				LinkedList<Integer> unexploredFirstChildClusterPoints = null;
				int numChildClusters = 0;

				
				while (!examinedVertices.isEmpty()) {
					TreeSet<Integer> constructingSubCluster = new TreeSet<Integer>();
					LinkedList<Integer> unexploredSubClusterPoints = new LinkedList<Integer>();
					boolean anyEdges = false;
					boolean incrementedChildCount = false;

					int rootVertex = examinedVertices.last();
					constructingSubCluster.add(rootVertex);
					unexploredSubClusterPoints.add(rootVertex);
					examinedVertices.remove(rootVertex);

					// Explore this potential child cluster as long as there are
					// unexplored points:
					while (!unexploredSubClusterPoints.isEmpty()) {
						int vertexToExplore = unexploredSubClusterPoints.poll();

						for (int neighbor : mst
								.getEdgeListForVertex2(vertexToExplore)) {
							anyEdges = true;

							if (constructingSubCluster.add(neighbor)) {
								unexploredSubClusterPoints.add(neighbor);
								examinedVertices.remove(neighbor);
							}
						}

						// Check if this potential child cluster is a valid
						// cluster:
						if (!incrementedChildCount && constructingSubCluster
								.size() >= minClusterSize && anyEdges) {
							incrementedChildCount = true;
							numChildClusters++;

							// If this is the first valid child cluster, stop
							// exploring it:
							if (firstChildCluster == null) {
								firstChildCluster = constructingSubCluster;
								unexploredFirstChildClusterPoints = unexploredSubClusterPoints;
								break;
							}
						}
					}

					// If there could be a split, and this child cluster is
					// valid:
					if (numChildClusters >= 2
							&& constructingSubCluster.size() >= minClusterSize
							&& anyEdges) {

						// Check this child cluster is not equal to the
						// unexplored first child cluster:
						int firstChildClusterMember = firstChildCluster.last();
						if (constructingSubCluster
								.contains(firstChildClusterMember)) {
							numChildClusters--;
						} // Otherwise, create a new cluster:
						else {
							Cluster newCluster = createNewClusterBubbles(
									constructingSubCluster,
									currentClusterLabels,
									clusters.get(examinedClusterLabel),
									nextClusterLabel, currentEdgeWeight, n);
							newClusters.add(newCluster);
							clusters.add(newCluster);
							nextClusterLabel++;
						}
					} // If this child cluster is not valid cluster, assign it
						// to noise:
					else if (constructingSubCluster.size() < minClusterSize
							|| !anyEdges) {
						createNewClusterBubbles(constructingSubCluster,
								currentClusterLabels,
								clusters.get(examinedClusterLabel), 0,
								currentEdgeWeight, n);

						for (int point : constructingSubCluster) {
							pointNoiseLevels[point] = currentEdgeWeight;
							pointLastClusters[point] = examinedClusterLabel;
						}
					}
				}

				// Finish exploring and cluster the first child cluster if there
				// was a split and it was not already clustered:
				if (numChildClusters >= 2
						&& currentClusterLabels[firstChildCluster
								.first()] == examinedClusterLabel) {

					while (!unexploredFirstChildClusterPoints.isEmpty()) {
						int vertexToExplore = unexploredFirstChildClusterPoints
								.poll();

						for (int neighbor : mst
								.getEdgeListForVertex2(vertexToExplore)) {
							if (firstChildCluster.add(neighbor)) {
								unexploredFirstChildClusterPoints.add(neighbor);
							}
						}
					}

					Cluster newCluster = createNewClusterBubbles(
							firstChildCluster, currentClusterLabels,
							clusters.get(examinedClusterLabel),
							nextClusterLabel, currentEdgeWeight, n);
					newClusters.add(newCluster);
					clusters.add(newCluster);
					nextClusterLabel++;
				}
			}

			// Write out the current level of the hierarchy:
			if (!compactHierarchy || nextLevelSignificant
					|| !newClusters.isEmpty()) {
				int outputLength = 0;

				String output = currentEdgeWeight + delimiter;
				hierarchyWriter.write(output);
				outputLength += output.length();

				for (int i = 0; i < previousClusterLabels.length - 1; i++) {
					output = previousClusterLabels[i] + delimiter;
					hierarchyWriter.write(output);
					outputLength += output.length();
				}

				output = previousClusterLabels[previousClusterLabels.length - 1]
						+ "\n";
				hierarchyWriter.write(output);
				outputLength += output.length();

				lineCount++;

				hierarchyCharsWritten += outputLength;
			}

			// Assign file offsets and calculate the number of constraints
			// satisfied:
			TreeSet<Integer> newClusterLabels = new TreeSet<Integer>();
			for (Cluster newCluster : newClusters) {
				newCluster.setFileOffset(hierarchyCharsWritten);
				newClusterLabels.add(newCluster.getLabel());
			}
			if (!newClusterLabels.isEmpty()) {
				calculateNumConstraintsSatisfiedBubbles(newClusterLabels,
						clusters, constraints, currentClusterLabels);
			}

			for (int i = 0; i < previousClusterLabels.length; i++) {
				previousClusterLabels[i] = currentClusterLabels[i];
			}

			if (newClusters.isEmpty()) {
				nextLevelSignificant = false;
			} else {
				nextLevelSignificant = true;
			}
		}

		// Write out the final level of the hierarchy (all points noise):
		hierarchyWriter.write(0 + delimiter);
		for (int i = 0; i < previousClusterLabels.length - 1; i++) {
			hierarchyWriter.write(0 + delimiter);
		}
		hierarchyWriter.write(0 + "\n");
		lineCount++;

		hierarchyWriter.close();

		Path pathClusterTree = new Path(inputFile + "/resultados/hdbscan/"
				+ countLevels + "/" + node + "/" + treeOutputFile);
		BufferedWriter treeWriter;
		treeWriter = new BufferedWriter(
				new OutputStreamWriter(fs.create(pathClusterTree, true)));

		// Write out the cluster tree:
		for (Cluster cluster : clusters) {
			if (cluster == null) {
				continue;
			}

			treeWriter.write(cluster.getLabel() + delimiter);
			treeWriter.write(cluster.getBirthLevel() + delimiter);
			treeWriter.write(cluster.getDeathLevel() + delimiter);
			treeWriter.write(cluster.getStability() + delimiter);

			if (constraints != null) {
				treeWriter.write((0.5 * cluster.getNumConstraintsSatisfied()
						/ constraints.size()) + delimiter);

				treeWriter.write(
						(0.5 * cluster.getPropagatedNumConstraintsSatisfied()
								/ constraints.size()) + delimiter);

			} else {
				treeWriter.write(0 + delimiter);

				treeWriter.write(0 + delimiter);

			}

			treeWriter.write(cluster.getFileOffset() + delimiter);

			if (cluster.getParent() != null) {
				treeWriter.write(cluster.getParent().getLabel() + "\n");

			} else {
				treeWriter.write(0 + "\n");
			}
		}
		String out = "";
		if (!compactHierarchy) {
			out = "1\n";

		} else {
			out = "0\n";
		}
		out = out + Integer.toString(lineCount);

		treeWriter.close();

		// interEdgesWriter.close();
		//
		// BufferedWriter visualizationWriter;
		// Path pathVisualization = new Path(visualizationOutputFile + "_" +
		// node);
		// visualizationWriter = new BufferedWriter(new OutputStreamWriter(
		// fs.create(pathVisualization, true)));
		// visualizationWriter.write(out);
		//
		// visualizationWriter.close();

		return clusters;
	}

	public boolean propagateTreeBubbles(ArrayList<Cluster> clusters) {
		TreeMap<Integer, Cluster> clustersToExamine = new TreeMap<Integer, Cluster>();
		BitSet addedToExaminationList = new BitSet(clusters.size());
		boolean infiniteStability = false;

		// Find all leaf clusters in the cluster tree:
		for (Cluster cluster : clusters) {
			if (cluster != null && !cluster.hasChildren()) {
				clustersToExamine.put(cluster.getLabel(), cluster);
				addedToExaminationList.set(cluster.getLabel());
			}
		}

		// Iterate through every cluster, propagating stability from children to
		// parents:
		while (!clustersToExamine.isEmpty()) {
			Cluster currentCluster = clustersToExamine.pollLastEntry()
					.getValue();
			currentCluster.propagate();

			if (currentCluster.getStability() == Double.POSITIVE_INFINITY) {
				infiniteStability = true;
			}

			if (currentCluster.getParent() != null) {
				Cluster parent = currentCluster.getParent();

				if (!addedToExaminationList.get(parent.getLabel())) {
					clustersToExamine.put(parent.getLabel(), parent);
					addedToExaminationList.set(parent.getLabel());
				}
			}
		}

		if (infiniteStability) {
			System.out.println(WARNING_MESSAGE);
		}
		return infiniteStability;
	}

	public int[] findProminentClustersBubbles(ArrayList<Cluster> clusters,
			String hierarchyFile, String flatOutputFile, String delimiter,
			int numPoints, boolean infiniteStability, String node,
			Configuration configuration, FileSystem fs, String inputFile)
			throws IOException, NumberFormatException {

		// Take the list of propagated clusters from the root cluster:
		ArrayList<Cluster> solution = clusters.get(1)
				.getPropagatedDescendants();

		// BufferedReader reader = new BufferedReader(
		// new FileReader(hierarchyFile));
		//
		// Configuration configuration = new Configuration();
		// configuration.addResource(new Path(
		// "/usr/local/hadoop/etc/hadoop/core-site.xml"));
		// configuration.addResource(new Path(
		// "/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));
		// FileSystem hdfsFileSystem = FileSystem.get(configuration);

		Path pathHierarchy = new Path(inputFile + "/resultados/hdbscan/"
				+ countLevels + "/" + node + "/" + hierarchyFile);
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(fs.open(pathHierarchy)));

		Path pathInter = new Path(inputFile + "/resultados/mst/" + countLevels
				+ "_" + node + "_/part-00000");
		BufferedWriter interWriter;
		interWriter = new BufferedWriter(
				new OutputStreamWriter(fs.create(pathInter, true)));

		int[] flatPartitioning = new int[numPoints];
		long currentOffset = 0;

		// Store all the file offsets at which to find the birth points for the
		// flat clustering:
		TreeMap<Long, ArrayList<Integer>> significantFileOffsets = new TreeMap<Long, ArrayList<Integer>>();
		for (Cluster cluster : solution) {
			ArrayList<Integer> clusterList = significantFileOffsets
					.get(cluster.getFileOffset());

			if (clusterList == null) {
				clusterList = new ArrayList<Integer>();
				significantFileOffsets.put(cluster.getFileOffset(),
						clusterList);
			}

			interWriter.write(0 + " " + 0 + " " + cluster.getBirthLevel() + " "
					+ 0 + " " + 0 + " " + node + "\n");
			clusterList.add(cluster.getLabel());
		}

		interWriter.close();
		// Go through the hierarchy file, setting labels for the flat
		// clustering:
		while (!significantFileOffsets.isEmpty()) {
			Map.Entry<Long, ArrayList<Integer>> entry = significantFileOffsets
					.pollFirstEntry();
			ArrayList<Integer> clusterList = entry.getValue();
			Long offset = entry.getKey();

			reader.skip(offset - currentOffset);
			String line = reader.readLine();

			currentOffset = offset + line.length() + 1;
			String[] lineContents = line.split(delimiter);

			for (int i = 1; i < lineContents.length; i++) {
				int label = Integer.parseInt(lineContents[i]);
				if (clusterList.contains(label)) {
					flatPartitioning[i - 1] = label;
				}
			}
		}
		reader.close();
		// Output the flat clustering result:
		Path pathFlat = new Path(inputFile + "/resultados/hdbscan/"
				+ countLevels + "/" + node + "/" + flatOutputFile);
		BufferedWriter writer;
		writer = new BufferedWriter(
				new OutputStreamWriter(fs.create(pathFlat, true)));

		if (infiniteStability) {
			writer.write(WARNING_MESSAGE + "\n");
		}

		for (int i = 0; i < flatPartitioning.length - 1; i++) {
			writer.write(flatPartitioning[i] + delimiter);
		}
		writer.write(flatPartitioning[flatPartitioning.length - 1] + "\n");
		writer.close();

		return flatPartitioning;
	}

	public ArrayList<OutlierScore> calculateOutlierScoresBubbles(
			ArrayList<Cluster> clusters, double[] pointNoiseLevels,
			int[] pointLastClusters, double[] coreDistances,
			String outlierScoresOutputFile, String delimiter,
			boolean infiniteStability, char node) throws IOException {

		int numPoints = pointNoiseLevels.length;
		ArrayList<OutlierScore> outlierScores = new ArrayList<OutlierScore>(
				numPoints);

		// Iterate through each point, calculating its outlier score:
		for (int i = 0; i < numPoints; i++) {
			double epsilon_max = clusters.get(pointLastClusters[i])
					.getPropagatedLowestChildDeathLevel();
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
		BufferedWriter writer = new BufferedWriter(
				new FileWriter(outlierScoresOutputFile + "_" + node),
				FILE_BUFFER_SIZE);
		if (infiniteStability) {
			writer.write(WARNING_MESSAGE + "\n");
		}

		for (OutlierScore outlierScore : outlierScores) {
			writer.write(outlierScore.getScore() + delimiter
					+ outlierScore.getId() + "\n");
		}
		writer.close();

		return outlierScores;
	}

	// ------------------------------ PRIVATE METHODS
	// ------------------------------
	private Cluster createNewClusterBubbles(TreeSet<Integer> points,
			int[] clusterLabels, Cluster parentCluster, int clusterLabel,
			double edgeWeight, int[] n) {
		int countMembers = 0;
		for (int point : points) {
			clusterLabels[point] = clusterLabel;
			countMembers += n[point];
		}
		parentCluster.detachPoints(points.size(), countMembers, edgeWeight);

		if (clusterLabel != 0) {
			return new Cluster(clusterLabel, parentCluster, edgeWeight,
					points.size());
		} else {
			parentCluster.addPointsToVirtualChildCluster(points);
			return null;
		}
	}

	private static void calculateNumConstraintsSatisfiedBubbles(
			TreeSet<Integer> newClusterLabels, ArrayList<Cluster> clusters,
			ArrayList<Constraint> constraints, int[] clusterLabels) {

		if (constraints == null) {
			return;
		}

		ArrayList<Cluster> parents = new ArrayList<Cluster>();
		for (int label : newClusterLabels) {
			Cluster parent = clusters.get(label).getParent();
			if (parent != null && !parents.contains(parent)) {
				parents.add(parent);
			}
		}

		for (Constraint constraint : constraints) {
			int labelA = clusterLabels[constraint.getPointA()];
			int labelB = clusterLabels[constraint.getPointB()];

			if (constraint.getType() == CONSTRAINT_TYPE.MUST_LINK
					&& labelA == labelB) {
				if (newClusterLabels.contains(labelA)) {
					clusters.get(labelA).addConstraintsSatisfied(2);
				}
			} else if (constraint.getType() == CONSTRAINT_TYPE.CANNOT_LINK
					&& (labelA != labelB || labelA == 0)) {
				if (labelA != 0 && newClusterLabels.contains(labelA)) {
					clusters.get(labelA).addConstraintsSatisfied(1);
				}
				if (labelB != 0 && newClusterLabels.contains(labelB)) {
					clusters.get(labelB).addConstraintsSatisfied(1);
				}

				if (labelA == 0) {
					for (Cluster parent : parents) {
						if (parent.virtualChildClusterContaintsPoint(
								constraint.getPointA())) {
							parent.addVirtualChildConstraintsSatisfied(1);
							break;
						}
					}
				}

				if (labelB == 0) {
					for (Cluster parent : parents) {
						if (parent.virtualChildClusterContaintsPoint(
								constraint.getPointB())) {
							parent.addVirtualChildConstraintsSatisfied(1);
							break;
						}
					}
				}
			}
		}

		for (Cluster parent : parents) {
			parent.releaseVirtualChildCluster();
		}
	}

	public double distanceBubbles(double distance,
			LinkedList<ClusterFeatureDataBubbles> bubbles, int point,
			int neighbor) {
		double verify = distance - (bubbles.get(point).getExtent()
				+ bubbles.get(neighbor).getExtent());
		if (verify >= 0) {
			distance = (distance - (bubbles.get(point).getExtent()
					+ bubbles.get(neighbor).getExtent()))
					+ (bubbles.get(point).getNnDist()
							+ bubbles.get(neighbor).getNnDist());
		} else {
			distance = Math.max(bubbles.get(point).getNnDist(),
					bubbles.get(neighbor).getNnDist());
		}
		return distance;
	}

	public Integer getProcessingU() {
		return processingU;
	}

	public void setProcessingU(Integer processingU) {
		this.processingU = processingU;
	}

	public DistanceCalculator getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(DistanceCalculator distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	public Integer getMinPoints() {
		return minPoints;
	}

	public void setMinPoints(Integer minPoints) {
		this.minPoints = minPoints;
	}

	public Integer getMinClusterSize() {
		return minClusterSize;
	}

	public void setMinClusterSize(Integer minClusterSize) {
		this.minClusterSize = minClusterSize;
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}

	public String getInputFile() {
		return inputFile;
	}

	public void setInputFile(String inputFile) {
		this.inputFile = inputFile;
	}

	public String getConstraintsFile() {
		return constraintsFile;
	}

	public void setConstraintsFile(String constraintsFile) {
		this.constraintsFile = constraintsFile;
	}

	public Integer getK() {
		return k;
	}

	public void setK(Integer k) {
		this.k = k;
	}

	public boolean isCompactHierarchy() {
		return compactHierarchy;
	}

	public void setCompactHierarchy(boolean compactHierarchy) {
		this.compactHierarchy = compactHierarchy;
	}

	public Integer getProcessing_units() {
		return processing_units;
	}

	public void setProcessing_units(Integer processing_units) {
		this.processing_units = processing_units;
	}

	public String getHierarchyFile() {
		return hierarchyFile;
	}

	public void setHierarchyFile(String hierarchyFile) {
		this.hierarchyFile = hierarchyFile;
	}

	public String getInterEdgesFile() {
		return interEdgesFile;
	}

	public void setInterEdgesFile(String interEdgesFile) {
		this.interEdgesFile = interEdgesFile;
	}

	public String getClusterTreeFile() {
		return clusterTreeFile;
	}

	public void setClusterTreeFile(String clusterTreeFile) {
		this.clusterTreeFile = clusterTreeFile;
	}

	public String getPartitionFile() {
		return partitionFile;
	}

	public void setPartitionFile(String partitionFile) {
		this.partitionFile = partitionFile;
	}

	public String getOutlierScoreFile() {
		return outlierScoreFile;
	}

	public void setOutlierScoreFile(String outlierScoreFile) {
		this.outlierScoreFile = outlierScoreFile;
	}

	public String getVisualizationFile() {
		return visualizationFile;
	}

	public void setVisualizationFile(String visualizationFile) {
		this.visualizationFile = visualizationFile;
	}

	public int getCountLevels() {
		return countLevels;
	}

	public void setCountLevels(int countLevels) {
		this.countLevels = countLevels;
	}
}*/
