package main;

import java.util.Map;

import org.apache.spark.api.java.function.Function2;

import databubbles.HdbscanDataBubbles;
import distance.DistanceCalculator;
import hdbscanstar.UndirectedGraph;
import scala.Tuple2;
import scala.Tuple3;

public class LocalModelReduceByKey implements
		Function2<Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>, Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>> {

	private static final long serialVersionUID = 1L;
	private Integer mpts;
	private Integer mclSize;
	private DistanceCalculator distanceFunction;
	private Map<Integer, Integer> key;

	public LocalModelReduceByKey(Integer mpts, Integer mclSize, DistanceCalculator dist, Map<Integer, Integer> key) {
		this.mpts = mpts;
		this.mclSize = mclSize;
		this.distanceFunction = dist;
		this.key = key;
	}

	public Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>> call(
			Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>> v1,
			Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>> v2)
			throws Exception {
		int[][] flatPartition = null;
		int count = 0;
		int id = v1._2()._3().intValue(); //=0
		int size = 0;
		//System.out.println("id: " + id);
		for (Integer key : this.key.keySet()) {  //(0,采样后的样本数量)
			if (id == key.intValue()) {
				size = this.key.get(key).intValue();
				//System.out.println("id="+id+",key="+key.intValue()+",size="+size);
			}
			//System.out.println(this.key);
		}
		double[][] data = new double[v1._1()._1().length + v2._1()._1().length][v2._1()._1()[0].length];
		double[][] infoBubbles = new double[v1._1()._2().length + v2._1()._2().length][v2._1()._2()[0].length];
		int[] idBubbles = new int[v1._1()._3().length + v2._1()._3().length];

		//data=[v1.rep;v2.rep]
		for (int i = 0; i < v1._1()._1().length; i++) {
			data[i] = v1._1()._1()[i];
			count++;
		}
		for (int i = 0; i < v2._1()._1().length; i++) {
			data[count++] = v2._1()._1()[i];
		}
		//infoBubbles=[[v1.extent,v1.nnDist,v1.n];[v2.extent,v2.nnDist,v2.n]]
		count = 0;
		for (int i = 0; i < v1._1()._2().length; i++) {
			infoBubbles[i] = v1._1()._2()[i];
			count++;
		}
		for (int i = 0; i < v2._1()._2().length; i++) {
			infoBubbles[count++] = v2._1()._2()[i];
		}
		//idBubbles=[v1.最近对象的序号,v2.最近对象的序号]
		count = 0;
		for (int i = 0; i < v1._1()._3().length; i++) {
			idBubbles[i] = v1._1()._3()[i];
			count++;
		}
		for (int i = 0; i < v2._1()._3().length; i++) {
			idBubbles[count++] = v2._1()._3()[i];
		}

		double[] eB = new double[infoBubbles.length];
		double[] nnDistB = new double[infoBubbles.length];
		int[] nB = new int[infoBubbles.length];

		for (int i = 0; i < eB.length; i++) {
			eB[i] = infoBubbles[i][0];
			nnDistB[i] = infoBubbles[i][1];
			nB[i] = (int) infoBubbles[i][2];
		}
		// computing hierarchy;
		//System.out.println(nB.length + " s: " + size);
		//System.out.println("nB.length="+nB.length+",size="+size);
		if (nB.length >= size) {
			HdbscanDataBubbles model = new HdbscanDataBubbles();
			double[] coreDistances = model.calculateCoreDistancesBubbles(data, nB, eB, nnDistB, this.mpts,
					this.distanceFunction);
			UndirectedGraph mst = model.constructMSTBubbles(data, nB, eB, nnDistB, idBubbles, coreDistances, true,
					this.distanceFunction);
			mst.quicksortByEdgeWeight();
			model.constructClusterTree(mst, this.mclSize, nB);
			flatPartition = model.findProminentClustersAndClassificationNoiseBubbles(model.getClusters(), data, nB, eB,
					nnDistB, this.distanceFunction, idBubbles);
			model.findInterClusterEdges(mst, flatPartition);
			int[] objects = new int[flatPartition.length];
			int[] labels = new int[flatPartition.length];
			for (int i = 0; i < labels.length; i++) {
				objects[i] = flatPartition[i][0];
				labels[i] = flatPartition[i][1];
			}
			return new Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>(
					new Tuple3<double[][], double[][], int[]>(data, infoBubbles, idBubbles),
					new Tuple3<int[], int[], Integer>(objects, labels, id),
					new Tuple3<int[], int[], double[]>(model.getVertice1(), model.getVertice2(), model.getDmreach()));
		} else {
			return new Tuple3<Tuple3<double[][], double[][], int[]>, Tuple3<int[], int[], Integer>, Tuple3<int[], int[], double[]>>(
					new Tuple3<double[][], double[][], int[]>(data, infoBubbles, idBubbles),
					new Tuple3<int[], int[], Integer>(null, null, id), null);
		}
	}

}
