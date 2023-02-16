package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;

import datastructure.DataPoints;
import distance.CosineSimilarity;
import distance.DistanceCalculator;
import distance.EuclideanDistance;
import distance.ManhattanDistance;
import distance.PearsonCorrelation;
import distance.SupremumDistance;
import hdbscanstar.UndirectedGraph;
import mappers.CombineStep;
import mappers.FirstStep;
import mappers.MapperDataset_github;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.Accumulator;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.LongAccumulator;
import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Array;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

import java.util.List;

public final class Main {

	private static final String save_path="/result/";
	private static final String FILE_FLAG = "file=";
	private static final String CLUSTERNAME_FLAG = "clusterName=";
	private static final String CONSTRAINTS_FLAG = "constraints=";
	private static final String MIN_PTS_FLAG = "minPts=";
	private static final String K_FLAG = "k=";
	private static final String PROCESSING_UNITS = "processing_units=";
	private static final String MIN_CL_SIZE_FLAG = "minClSize=";
	private static final String COMPACT_FLAG = "compact=";
	private static final String DISTANCE_FUNCTION_FLAG = "dist_function=";

	private static final String EUCLIDEAN_DISTANCE = "euclidean";
	private static final String COSINE_SIMILARITY = "cosine";
	private static final String PEARSON_CORRELATION = "pearson";
	private static final String MANHATTAN_DISTANCE = "manhattan";
	private static final String SUPREMUM_DISTANCE = "supremum";

	private static SparkConf conf;
	private static JavaRDD<String> dataFile;
	private static JavaPairRDD<Integer, Tuple2<Integer, double[]>> dataset;
	// public static int nodeCount = 0;
	public static LongAccumulator newIteration;

	public static void main(String[] args) throws Exception {

		String[] my_args=new String[]{"file=dataset.txt","minPts=4","minClSize=4","compact=true","processing_units=50","k=0.2","clusterName=local"};
		// Parse input parameters from program arguments:
		HDBSCANStarParameters parameters = checkInputParameters(my_args);

		System.out.println("Running MR-HDBSCAN* on " + parameters.inputFile + " with minPts=" + parameters.minPoints
				+ ", minClSize=" + parameters.minClusterSize + ", dist_function=" + parameters.distanceFunction
				+ ", processing_units=" + parameters.processing_units + ", k=" + parameters.k + ", clusterName="
				+ parameters.clusterName);

		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		// Creating a JavaSparkContext
		String inputName = parameters.inputFile;
		if (parameters.inputFile.contains(".")) {
			inputName = parameters.inputFile.substring(0, parameters.inputFile.lastIndexOf("."));
		}

		conf = new SparkConf().setAppName("MR-HDBSCAN* on dataset: " + inputName).setMaster(parameters.clusterName) // spark://master:7077
				.set("spark.hadoop.validateOutputSpecs", "false");
		/*
		 * .set("spark.driver.cores", "6") .set("spark.driver.memory", "6g")
		 * .set("spark.executor.memory", "6g");
		 */
		JavaSparkContext jsc = new JavaSparkContext(conf);
		// modificar isso aqui para algo generalizado
		dataFile = jsc.textFile(parameters.inputFile);
		System.out.println(parameters.inputFile);
		// reading dataset and saving each object as a pair <id, object>
		dataset = dataFile.mapToPair(new MapperDataset_github()).cache();
		dataset.coalesce(1, true).saveAsObjectFile(parameters.inputFile + "_unprocessed_0");
		System.out.println(parameters.inputFile + "_unprocessed_0");
		int iteration = 0, processedPointsCounter = 0, splitByKey = 1;
		int datasetSize = (int) dataset.count();
		int nextIdSubsets = 2;
		// first step of MR-HDBSCAN* (using a mapPartitionsToPair)
		while (processedPointsCounter < datasetSize) {
			System.out.println("Iteration: " + iteration);
			JavaRDD<Tuple2<Integer, Tuple2<Integer, double[]>>> data;
			data = jsc.objectFile(parameters.inputFile + "_unprocessed_" + iteration);
			// compute the size of each subset and save the number of attributes of each object using the keys
			List<Tuple2<Integer, Tuple2<Integer, Integer>>> keyCounts = JavaPairRDD.fromJavaRDD(data).mapToPair(
					new PairFunction<Tuple2<Integer, Tuple2<Integer, double[]>>, Integer, Tuple2<Integer, Integer>>() {
						private static final long serialVersionUID = 1L;

						public Tuple2<Integer, Tuple2<Integer, Integer>> call(
								Tuple2<Integer, Tuple2<Integer, double[]>> t) throws Exception {
							return new Tuple2<Integer, Tuple2<Integer, Integer>>(t._1(),
									new Tuple2<Integer, Integer>(1, t._2()._2().length));
						}
					}).reduceByKey(
							new Function2<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>, Tuple2<Integer, Integer>>() {
								private static final long serialVersionUID = 1L;

								public Tuple2<Integer, Integer> call(Tuple2<Integer, Integer> v1,
										Tuple2<Integer, Integer> v2) throws Exception {
									return new Tuple2<Integer, Integer>(v1._1().intValue() + v2._1().intValue(),
											v1._2());
								}
							})
					.collect();
			Map<Integer, Double> fractions = new HashMap<Integer, Double>();
			for (int i = 0; i < keyCounts.size(); i++) {
				fractions.put(keyCounts.get(i)._1(), parameters.k);
				if (keyCounts.get(i)._2()._1().intValue() <= parameters.processing_units) {
					processedPointsCounter += keyCounts.get(i)._2()._1().intValue();
				}
			}
			// selecting stratified sampling from multiple subsets
			List<Tuple2<Integer, Tuple2<Integer, double[]>>> s = JavaPairRDD.fromJavaRDD(data)
					.sampleByKeyExact(false, fractions).collect();

			List<Tuple2<Integer, Tuple2<Integer, double[]>>> samples = new ArrayList<Tuple2<Integer, Tuple2<Integer, double[]>>>();

			for (int i = 0; i < keyCounts.size(); i++) {
				int count = 0;
				for (int j = 0; j < s.size(); j++) {
					if (keyCounts.get(i)._1().intValue() == s.get(j)._1().intValue()) {
						samples.add(new Tuple2<Integer, Tuple2<Integer, double[]>>(s.get(j)._1(),
								new Tuple2<Integer, double[]>(count, s.get(j)._2()._2())));
						count++;
					}
				}
			}

			Map<Integer, Integer> countSamplesByKey = new HashMap<Integer, Integer>();
			for (int i = 0; i < samples.size(); i++) {
				if (countSamplesByKey.get(samples.get(i)._1().intValue()) == null) {
					countSamplesByKey.put(samples.get(i)._1().intValue(), 0);
				}
				int value = countSamplesByKey.get(samples.get(i)._1().intValue());
				countSamplesByKey.put(samples.get(i)._1().intValue(), value + 1);
			}

			// doing the first step of MR-HDBSCAN*
			JavaPairRDD<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> subsets = JavaPairRDD
					.fromJavaRDD(data)
					.mapPartitionsToPair(new FirstStep(parameters.k, parameters.processing_units, parameters.minPoints,
							parameters.distanceFunction, parameters.inputFile, iteration, samples, keyCounts));

			// mapping the edges...
			JavaPairRDD<Integer, Tuple3<Integer, Integer, Double>> mst = subsets.filter(
					new Function<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>, Boolean>() {
						private static final long serialVersionUID = 1L;

						public Boolean call(
								Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> arg0)
								throws Exception {
							return arg0._1().intValue() == -1;
						}
					}).flatMapToPair(
							new PairFlatMapFunction<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>, Integer, Tuple3<Integer, Integer, Double>>() {
								private static final long serialVersionUID = 1L;
								private int count = 0;

								public Iterator<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> call(
										Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> t)
										throws Exception {
									ArrayList<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> mst = new ArrayList<Tuple2<Integer, Tuple3<Integer, Integer, Double>>>();
									int v = t._2()._1();
									int u = t._2()._3();
									double dmreach = t._2()._2()._1()[0];
									// 1, v, dmreach, u
									mst.add(new Tuple2<Integer, Tuple3<Integer, Integer, Double>>(-1,
											new Tuple3<Integer, Integer, Double>(v, u, dmreach)));
									return mst.iterator();
								}
							});
			mst.saveAsObjectFile(parameters.inputFile + "_local_mst" + iteration);
			System.out.println(parameters.inputFile + "_local_mst" + iteration);
			iteration++;
			if (processedPointsCounter < datasetSize) {
				subsets = subsets.filter(
						new Function<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>, Boolean>() {
							private static final long serialVersionUID = 1L;

							public Boolean call(
									Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> arg0)
									throws Exception {
								return arg0._1().intValue() != -1;
							}
						});
				JavaPairRDD<Integer, Tuple3<Integer, Integer, double[]>> tes = subsets.mapToPair( // nearest,
																									// id,
																									// idobject,
																									// object
						new PairFunction<Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>>, Integer, Tuple3<Integer, Integer, double[]>>() {
							private static final long serialVersionUID = 1L;

							public Tuple2<Integer, Tuple3<Integer, Integer, double[]>> call(
									Tuple2<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> t)
									throws Exception {
								// nearest, <id, idObject, object>
								return new Tuple2<Integer, Tuple3<Integer, Integer, double[]>>(t._1(),
										new Tuple3<Integer, Integer, double[]>(t._2()._3(), t._2()._1(),
												t._2()._2()._1()));
							}

						});
				tes.saveAsObjectFile(parameters.inputFile + "_nearest_" + iteration);
				tes.saveAsTextFile(parameters.inputFile + "_bullshit_" + iteration);
				System.out.println(parameters.inputFile + "_nearest_" + iteration);
				System.out.println(parameters.inputFile + "_bullshit_" + iteration);

				// local and global combining (creating Data Bubbles);
				JavaPairRDD<Integer, Tuple3<Integer, Tuple4<double[], double[], double[], double[]>, Integer>> dataBubbles = subsets
						.reduceByKey(new CombineStep());
				dataBubbles.saveAsTextFile(parameters.inputFile + "_data_bubbles_" + iteration);
				System.out.println(parameters.inputFile + "_data_bubbles_" + iteration);

				// HDBSCAN* hierarchy from data bubbles
				// id, bubbles, partition from bubbles, inter-cluster edges;
				JavaPairRDD<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>> localModel = dataBubbles
						.mapToPair(new ReMapBubbles()).reduceByKey(new LocalModelReduceByKey(parameters.minPoints,
								parameters.minClusterSize, parameters.distanceFunction, countSamplesByKey));

				// filtering and persisting the inter-cluster edges
				localModel.flatMapToPair(
						new PairFlatMapFunction<Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>>, Integer, Tuple3<Integer, Integer, Double>>() {
							private static final long serialVersionUID = 1L;

							public Iterator<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> call(
									Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>> t)
									throws Exception {
								ArrayList<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> mst = new ArrayList<Tuple2<Integer, Tuple3<Integer, Integer, Double>>>();
								if (t._2()._3() != null) {
									// 1, v, dmreach, u
									if (t._2()._3()._2() != null)
										mst.add(new Tuple2<Integer, Tuple3<Integer, Integer, Double>>(-1,
												new Tuple3<Integer, Integer, Double>(t._2()._3()._1()[0],
														t._2()._3()._2()[0], t._2()._3()._3()[0])));
								}
								return mst.iterator();
							}
						}).saveAsObjectFile(parameters.inputFile + "_local_mst_" + iteration);
				System.out.println(parameters.inputFile + "_local_mst_" + iteration);

				List<Tuple2<Integer, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>>> model = localModel
						.collect();

				// data partition induction
				splitByKey = 0;
				TreeSet<Integer> newIds = new TreeSet<Integer>();
				for (int i = 0; i < model.size(); i++) {
					//System.out.println(model.get(i));
					for (int member = 0; member < model.get(i)._2()._2()._1().length; member++) {
						newIds.add(model.get(i)._2()._2()._2()[member]);
					}
					splitByKey += newIds.size();
					while (!newIds.isEmpty()) {
						int clusterId = newIds.pollFirst();
						for (int member = 0; member < model.get(i)._2()._2()._1().length; member++) {
							if (model.get(i)._2()._2()._2()[member] == clusterId) {
								model.get(i)._2()._2()._2()[member] = nextIdSubsets;
							}
						}
						nextIdSubsets++;
					}
				}
				splitByKey = (splitByKey > 0) ? splitByKey : 1;
				// reading subsets
				JavaRDD<Tuple2<Integer, Tuple3<Integer, Integer, double[]>>> data1 = jsc
						.objectFile(parameters.inputFile + "_nearest_" + iteration);

				// JavaPairRDD<Integer, Tuple2<Integer, double[]>> e =
				JavaPairRDD.fromJavaRDD(data1).mapToPair(new LabelClassification(model))
						.partitionBy(new HashPartitioner(splitByKey))
						.saveAsObjectFile(parameters.inputFile + "_unprocessed_" + iteration);
				System.out.println(parameters.inputFile + "_unprocessed_" + iteration);
			}
		}
		// second step of MR-HDBSCAN*
		// merging and sorting the local MSTs in an unique solution;
		JavaRDD<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> MSTFiles = jsc
				.objectFile(parameters.inputFile + "_local_mst*");

		int count = (int) JavaPairRDD.fromJavaRDD(MSTFiles)
				.mapToPair(new PairFunction<Tuple2<Integer, Tuple3<Integer, Integer, Double>>, Integer, Integer>() {
					private static final long serialVersionUID = 1L;

					public Tuple2<Integer, Integer> call(Tuple2<Integer, Tuple3<Integer, Integer, Double>> t)
							throws Exception {
						return new Tuple2<Integer, Integer>(1, 1);
					}
				}).count();
		System.out.println("Tamanho: " + count);

		JavaPairRDD.fromJavaRDD(MSTFiles).flatMapToPair(
				new PairFlatMapFunction<Tuple2<Integer, Tuple3<Integer, Integer, Double>>, Integer, int[]>() {
					private static final long serialVersionUID = 1L;

					public Iterator<Tuple2<Integer, int[]>> call(Tuple2<Integer, Tuple3<Integer, Integer, Double>> t)
							throws Exception {
						ArrayList<Tuple2<Integer, int[]>> list = new ArrayList<Tuple2<Integer, int[]>>();
						int[] v = { t._2()._2() };
						int[] u = { t._2()._1() };
						list.add(new Tuple2<Integer, int[]>(t._2()._1(), v));
						list.add(new Tuple2<Integer, int[]>(t._2()._2(), u));
						return list.iterator();
					}
				}).reduceByKey(new Function2<int[], int[], int[]>() {
					private static final long serialVersionUID = 1L;

					public int[] call(int[] v1, int[] v2) throws Exception {
						int[] adj = new int[v1.length + v2.length];
						int count = 0;
						for (int i = 0; i < v1.length; i++) {
							adj[count] = v1[i];
							count++;
						}
						for (int i = 0; i < v2.length; i++) {
							adj[count] = v2[i];
							count++;
						}
						return adj;
					}
				}).saveAsObjectFile(parameters.inputFile + "_adjList");
		System.out.println(parameters.inputFile + "_adjList");

		int numberOfEdges = ((datasetSize * 2) - 1);
		while (numberOfEdges >= 0) {
			JavaPairRDD<Integer, Tuple3<Integer, Integer, Double>> m = JavaPairRDD.fromJavaRDD(MSTFiles).mapToPair(
					new PairFunction<Tuple2<Integer, Tuple3<Integer, Integer, Double>>, Integer, Tuple3<Integer, Integer, Double>>() {
						private static final long serialVersionUID = 1L;

						public Tuple2<Integer, Tuple3<Integer, Integer, Double>> call(
								Tuple2<Integer, Tuple3<Integer, Integer, Double>> t) throws Exception {
							return new Tuple2<Integer, Tuple3<Integer, Integer, Double>>(t._1(),
									new Tuple3<Integer, Integer, Double>(t._2()._1(), t._2()._2(), t._2()._3()));
						}
					});

			List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> highestEdgeWeight = m.reduceByKey(
					new Function2<Tuple3<Integer, Integer, Double>, Tuple3<Integer, Integer, Double>, Tuple3<Integer, Integer, Double>>() {
						private static final long serialVersionUID = 1L;

						public Tuple3<Integer, Integer, Double> call(Tuple3<Integer, Integer, Double> v1,
								Tuple3<Integer, Integer, Double> v2) throws Exception {
							return (v1._3() >= v2._3()) ? v1 : v2;
						}
					}).collect();
			// filtering the tied highest edges on the current hierarchy level.
			List<Tuple2<Integer, Tuple3<Integer, Integer, Double>>> higher = m
					.filter(new FilterTiedEdges(highestEdgeWeight)).collect();

			JavaRDD<Tuple2<Integer, int[]>> adj = jsc.objectFile(parameters.inputFile + "_adjList");
			JavaPairRDD<Integer, int[]> adjList = JavaPairRDD.fromJavaRDD(adj)
					.mapToPair(new FilterAdjacentVertex(higher));
			m.filter(new FilterHighestEdgeWeight(higher)).saveAsObjectFile(parameters.inputFile + "_newMST");
			System.out.println(parameters.inputFile + "_newMST");
			MSTFiles = jsc.objectFile(parameters.inputFile + "_newMST");

			System.out.println(" size: " + numberOfEdges);
			// finding connected components
			JavaPairRDD<Integer, int[]> subcomponents = adjList;
			do {
				newIteration = jsc.sc().longAccumulator(); // init with 0
				System.out.println("New iteration - 1:  " + newIteration.value());
				subcomponents = subcomponents.flatMapToPair(new findConnectedComponentsOnMST())
						.reduceByKey(new Function2<int[], int[], int[]>() {
							private static final long serialVersionUID = 1L;

							public int[] call(int[] v1, int[] v2) throws Exception {
								int[] newArray = new int[v1.length + v2.length];
								for (int i = 0; i < v1.length; i++) {
									newArray[i] = v1[i];
								}
								for (int i = 0; i < v2.length; i++) {
									newArray[v1.length + i] = v2[i];
								}
								return newArray;
							}
						});
				subcomponents.saveAsTextFile(parameters.inputFile + "_comp");
				System.out.println(parameters.inputFile + "_comp");
				System.out.println("New iteration? " + newIteration.value());
			} while (newIteration.value() != 0);
			System.exit(1);

			numberOfEdges -= higher.size();
		}

	}

	/* ANOTHER METHODS */

	private static HDBSCANStarParameters checkInputParameters(String[] args) {
		HDBSCANStarParameters parameters = new HDBSCANStarParameters();
		parameters.distanceFunction = new EuclideanDistance();
		parameters.compactHierarchy = false;

		// Read in the input arguments and assign them to variables:
		for (String argument : args) {

			// Assign input file:
			if (argument.startsWith(FILE_FLAG) && argument.length() > FILE_FLAG.length()) {
				parameters.inputFile = argument.substring(FILE_FLAG.length());
			}

			if (argument.startsWith(CLUSTERNAME_FLAG) && argument.length() > CLUSTERNAME_FLAG.length()) {
				parameters.clusterName = argument.substring(CLUSTERNAME_FLAG.length());
			}

			// Assign constraints file:
			if (argument.startsWith(CONSTRAINTS_FLAG) && argument.length() > CONSTRAINTS_FLAG.length()) {
				parameters.constraintsFile = argument.substring(CONSTRAINTS_FLAG.length());
			} // Assign minPoints:
			else if (argument.startsWith(MIN_PTS_FLAG) && argument.length() > MIN_PTS_FLAG.length()) {
				try {
					parameters.minPoints = Integer.parseInt(argument.substring(MIN_PTS_FLAG.length()));
				} catch (NumberFormatException nfe) {
					System.out.println("Illegal v.distanceFunction");
				}
			} else if (argument.startsWith(K_FLAG) && argument.length() > K_FLAG.length()) {
				try {
					parameters.k = Double.parseDouble(argument.substring(K_FLAG.length()));
				} catch (NumberFormatException nfe) {
					System.out.println("Illegal v.distanceFunction");
				}
			}

			else if (argument.startsWith(PROCESSING_UNITS) && argument.length() > PROCESSING_UNITS.length()) {
				try {
					parameters.processing_units = Integer.parseInt(argument.substring(PROCESSING_UNITS.length()));
				} catch (NumberFormatException nfe) {
					System.out.println("Illegal value for processing units");
				}
			}
			// Assign minClusterSize:
			else if (argument.startsWith(MIN_CL_SIZE_FLAG) && argument.length() > MIN_CL_SIZE_FLAG.length()) {
				try {
					parameters.minClusterSize = Integer.parseInt(argument.substring(MIN_CL_SIZE_FLAG.length()));
				} catch (NumberFormatException nfe) {
					System.out.println("Illegal value for minClSize.");
				}
			} // Assign compact hierarchy:
			else if (argument.startsWith(COMPACT_FLAG) && argument.length() > COMPACT_FLAG.length()) {
				parameters.compactHierarchy = Boolean.parseBoolean(argument.substring(COMPACT_FLAG.length()));

			} // Assign distance function:
			else if (argument.startsWith(DISTANCE_FUNCTION_FLAG)
					&& argument.length() > DISTANCE_FUNCTION_FLAG.length()) {
				String functionName = argument.substring(DISTANCE_FUNCTION_FLAG.length());

				if (functionName.equals(EUCLIDEAN_DISTANCE)) {
					parameters.distanceFunction = new EuclideanDistance();
				} else if (functionName.equals(COSINE_SIMILARITY)) {
					parameters.distanceFunction = new CosineSimilarity();
				} else if (functionName.equals(PEARSON_CORRELATION)) {
					parameters.distanceFunction = new PearsonCorrelation();
				} else if (functionName.equals(MANHATTAN_DISTANCE)) {
					parameters.distanceFunction = new ManhattanDistance();
				} else if (functionName.equals(SUPREMUM_DISTANCE)) {
					parameters.distanceFunction = new SupremumDistance();
				} else {
					parameters.distanceFunction = null;
				}
			}
		}

		// Check that each input parameter has been assigned:
		if (parameters.inputFile == null) {
			System.out.println("Missing input file name.");
			printHelpMessageAndExit();
		} else if (parameters.minPoints == null) {
			System.out.println("Missing value for minPts.");
			printHelpMessageAndExit();
		} else if (parameters.k == null) {
			System.out.println("Missing value for k.");
			printHelpMessageAndExit();
		} else if (parameters.minClusterSize == null) {
			System.out.println("Missing value for minClSize");
			printHelpMessageAndExit();
		} else if (parameters.distanceFunction == null) {
			System.out.println("Missing distance function.");
			printHelpMessageAndExit();
		}

		// // Generate names for output files:
		// String inputName = parameters.inputFile;
		// if (parameters.inputFile.contains(".")) {
		// inputName = parameters.inputFile.substring(0,
		// parameters.inputFile.lastIndexOf("."));
		// }

		if (parameters.compactHierarchy) {
			parameters.hierarchyFile = "base_compact_hierarchy.csv";
		} else {
			parameters.hierarchyFile = "base_hierarchy.csv";
		}
		parameters.clusterTreeFile = "base_tree.csv";
		parameters.partitionFile = "base_partition.csv";
		parameters.outlierScoreFile = "base_outlier_scores.csv";
		parameters.visualizationFile = "base_visualization.vis";
		parameters.interEdgesFile = "base_interEdges.csv";

		return parameters;
	}

	/**
	 * Prints a help message that explains the usage of HDBSCANStarRunner, and
	 * then exits the program.
	 */
	private static void printHelpMessageAndExit() {
		System.out.println();

		System.out.println("Executes the MR-HDBSCAN* algorithm, which produces a hierarchy, cluster tree, "
				+ "flat partitioning, and outlier scores for an input data set.");
		System.out.println("Usage: java -jar MR-HDBSCANStar.jar file=<input file> minPts=<minPts value> "
				+ "minClSize=<minClSize value> [constraints=<constraints file>] [compact={true,false}] "
				+ "[dist_function=<distance function>]");
		System.out.println("By default the hierarchy produced is non-compact (full), and euclidean distance is used.");
		System.out.println("Example usage: \"java -jar MR-HDBSCANStar.jar file=input.csv minPts=4 minClSize=4\"");
		System.out.println("Example usage: \"java -jar MR-HDBSCANStar.jar file=collection.csv minPts=6 minClSize=1 "
				+ "constraints=collection_constraints.csv dist_function=manhattan\"");
		System.out.println("Example usage: \"java -jar MR-HDBSCANStar.jar file=data_set.csv minPts=8 minClSize=8 "
				+ "compact=true\"");
		System.out.println("In cases where the source is compiled, use the following: \"java HDBSCANStarRunner "
				+ "file=data_set.csv minPts=8 minClSize=8 compact=true\"");
		System.out.println();

		System.out.println("The input data set file must be a comma-separated value (CSV) file, where each line "
				+ "represents an object, with attributes separated by commas.");
		System.out.println(
				"The algorithm will produce five files: the hierarchy, cluster tree, final flat partitioning, outlier scores, and an auxiliary file for visualization.");
		System.out.println();

		System.out.println("The hierarchy file will be named <input>_hierarchy.csv for a non-compact "
				+ "(full) hierarchy, and <input>_compact_hierarchy.csv for a compact hierarchy.");
		System.out.println("The hierarchy file will have the following format on each line:");
		System.out.println(
				"<hierarchy scale (epsilon radius)>,<label for object 1>,<label for object 2>,...,<label for object n>");
		System.out.println("Noise objects are labelled zero.");
		System.out.println();

		System.out.println("The cluster tree file will be named <input>_tree.csv");
		System.out.println("The cluster tree file will have the following format on each line:");
		System.out.println("<cluster label>,<birth level>,<death level>,<stability>,<gamma>,"
				+ "<virtual child cluster gamma>,<character_offset>,<parent>");
		System.out.println("<character_offset> is the character offset of the line in the hierarchy "
				+ "file at which the cluster first appears.");
		System.out.println();

		System.out.println("The final flat partitioning file will be named <input>_partition.csv");
		System.out.println("The final flat partitioning file will have the following format on a single line:");
		System.out.println("<label for object 1>,<label for object 2>,...,<label for object n>");
		System.out.println();

		System.out.println("The outlier scores file will be named <input>_outlier_scores.csv");
		System.out.println("The outlier scores file will be sorted from 'most inlier' to 'most outlier', "
				+ "and will have the following format on each line:");
		System.out.println("<outlier score>,<object id>");
		System.out.println("<object id> is the zero-indexed line on which the object appeared in the input file.");
		System.out.println();

		System.out.println("The auxiliary visualization file will be named <input>_visulization.vis");
		System.out.println("This file is only used by the visualization module.");
		System.out.println();

		System.out.println("The optional input constraints file can be used to provide constraints for "
				+ "the algorithm (semi-supervised flat partitioning extraction).");
		System.out.println("If this file is not given, only stability will be used to selected the "
				+ "most prominent clusters (unsupervised flat partitioning extraction).");
		System.out.println("This file must be a comma-separated value (CSV) file, where each line "
				+ "represents a constraint, with the two zero-indexed objects and type of constaint "
				+ "separated by commas.");
		System.out.println("Use 'ml' to specify a must-link constraint, and 'cl' to specify a cannot-link constraint.");
		System.out.println();

		System.out.println("The optional compact flag can be used to specify if the hierarchy saved to file "
				+ "should be the fu14878080ll or the compact one (this does not affect the final partitioning or cluster tree).");
		System.out.println("The full hierarchy includes all levels where objects change clusters or "
				+ "become noise, while the compact hierarchy only includes levels where clusters are born or die.");
		System.out.println();

		System.out.println("Possible values for the optional dist_function flag are:");
		System.out.println("euclidean: Euclidean Distance, d = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2)");
		System.out.println("cosine: Cosine Similarity, d = 1 - ((X�Y) / (||X||*||Y||))");
		System.out.println("pearson: Pearson Correlation, d = 1 - (cov(X,Y) / (std_dev(X) * std_dev(Y)))");
		System.out.println("manhattan: Manhattan Distance, d = |x1-y1| + |x2-y2| + ... + |xn-yn|");
		System.out.println("supremum: Supremum Distance, d = max[(x1-y1), (x2-y2), ... ,(xn-yn)]");
		System.out.println();

		System.exit(0);
	}

	/**
	 * Simple class for storing input parameters.
	 */
	private static class HDBSCANStarParameters {

		public String inputFile;
		public String constraintsFile;
		public Integer minPoints;
		public Double k;
		public Integer minClusterSize;
		public boolean compactHierarchy;
		public DistanceCalculator distanceFunction;
		public Integer processing_units;
		public String clusterName;

		public String hierarchyFile;
		public String interEdgesFile;
		public String clusterTreeFile;
		public String partitionFile;
		public String outlierScoreFile;
		public String visualizationFile;
	}
}
