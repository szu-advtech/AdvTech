package datastructure;

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
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import distance.DistanceCalculator;

public class DataBubbles implements Serializable {

	
	
	
	private static final long serialVersionUID = 1L;
	private int[] idB;
	private int[] id_cluster;
	private int[] nodesID;
	private int[] tamClusters;
	private double[][] rep;
	private double[] extent;
	private double[] nnDist;
	private int[] n;
	private String[] keys;
	private int numOfClusters;
    private double[] coreDistances;
	
	
	public DataBubbles() {

	}

	public DataBubbles(int[] id_cluster, int[] tam, int[] idB, double[][] rep,
			double[] extent, double[] nnDist, int[] n) {
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setN(n);
	}

	public DataBubbles(double[][] rep, double[] extent, double[] nnDist, int[] n) {
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setN(n);
	}

	public DataBubbles(int[] id_cluster, int[] tam, int[] idB) {
		this.setIdB(idB);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
	}

	public DataBubbles(String[] keys, int[] id_cluster, int[] tam, int[] idB) {
		this.setKeys(keys);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
	}

	public DataBubbles(String[] keys, int[] id_cluster, int[] tam, int[] idB, int numOfClusters) {
		this.setKeys(keys);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
		this.setNumOfClusters(numOfClusters);
	}

	public DataBubbles(int[] nodesID, int[] id_cluster, int[] tam, int[] idB,
			double[][] rep, double[] extent, double[] nnDist, int[] n) {
		this.setNodesID(nodesID);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setN(n);
	}

	public DataBubbles(String[] keys, int[] nodesID, int[] id_cluster, int[] tam, int[] idB, int numOfClusters) {
		this.setKeys(keys);
		this.setNodesID(nodesID);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
		this.setNumOfClusters(numOfClusters);
	}
	
	
	// ///////////////////////////////////////////////////////////////////////////////////////

	public DataBubbles(String[] keys, int[] nodesID, int[] id_cluster,
			int[] tam, int[] idB, int numOfClusters, double[] coreDistances, double[][] rep, double[] extent, double[] nnDist, int[] n) {
		
		this.setKeys(keys);
		this.setNodesID(nodesID);
		this.setIdCluster(id_cluster);
		this.setTamClusters(tam);
		this.setIdB(idB);
		this.setNumOfClusters(numOfClusters);
		this.setCoreDistances(coreDistances);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setN(n);
	}

	public void setIdB(int[] idB) {
		this.idB = idB;
	}

	public void setIdCluster(int[] id_cluster) {
		this.id_cluster = id_cluster;
	}

	public int[] getIdB() {
		return this.idB;
	}

	public int[] getIdCluster() {
		return this.id_cluster;
	}

	public double[][] getRep() {
		return rep;
	}

	public void setRep(double[][] rep) {
		this.rep = rep;
	}

	public double[] getExtent() {
		return extent;
	}

	public void setExtent(double[] extent) {
		this.extent = extent;
	}

	public double[] getNnDist() {
		return nnDist;
	}

	public void setNnDist(double[] nnDist) {
		this.nnDist = nnDist;
	}

	public int[] getN() {
		return n;
	}

	public void setN(int[] n) {
		this.n = n;
	}

	public int[] getTamClusters() {
		return tamClusters;
	}

	public void setTamClusters(int[] tamClusters) {
		this.tamClusters = tamClusters;
	}

	

	public String[] getKeys() {
		return keys;
	}

	public void setKeys(String[] keys) {
		this.keys = keys;
	}

	public int getNumOfClusters() {
		return numOfClusters;
	}

	public void setNumOfClusters(int numOfClusters) {
		this.numOfClusters = numOfClusters;
	}

	public int[] getNodesID() {
		return nodesID;
	}

	public void setNodesID(int[] nodesID) {
		this.nodesID = nodesID;
	}
    
	@Override
	public String toString() {
		return "DataBubbles [idB=" + Arrays.toString(idB) + ", id_cluster="
				+ Arrays.toString(id_cluster) + ", tamClusters="
				+ Arrays.toString(tamClusters) + "]";
	}

	public double[] getCoreDistances() {
		return coreDistances;
	}

	public void setCoreDistances(double[] coreDistances) {
		this.coreDistances = coreDistances;
	}
}
