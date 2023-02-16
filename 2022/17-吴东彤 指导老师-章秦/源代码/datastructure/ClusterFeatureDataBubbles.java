package datastructure;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;

public class ClusterFeatureDataBubbles implements Serializable {

	private static final long serialVersionUID = 1L;
	private int id;
	private double[] ls;
	private double[] ss;
	private int n;
	private double[] rep;
	private double extent;
	private double nnDist;
	private DataPoints points;
	private DataBubbles bubbles;
	private double distance;
	private String key;
	private LinkedList<ClusterFeatureDataBubbles> cfs;

	public ClusterFeatureDataBubbles() {
	}

	public ClusterFeatureDataBubbles(int id, double[] ls, double[] ss, int n,
			DataPoints points, DataBubbles bubbles) {
		this.setId(id);
		this.setLs(ls);
		this.setSs(ss);
		this.setN(n);
		this.setPoints(points);
		this.setBubbles(bubbles);
	}

	public ClusterFeatureDataBubbles(int id, double[] rep, double extent,
			double nnDist, int n, DataPoints points) {
		this.setId(id);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setN(n);
		this.setPoints(points);
	}

	public ClusterFeatureDataBubbles(String key, double[] ls, double[] ss,
			int nn, double[] rep, double extent, double nnDist) {
		this.setKey(key);
		this.setLs(ls);
		this.setSs(ss);
		this.setN(nn);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
	}

	public ClusterFeatureDataBubbles(int id, double[] ls, double[] ss, int n,
			double[] rep, double extent, double nnDist) {
		this.setId(id);
		this.setLs(ls);
		this.setSs(ss);
		this.setN(n);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
	}

	public ClusterFeatureDataBubbles(int id, double[] ls, double[] ss, int n,
			double[] rep, double extent, double nnDist, DataPoints points,
			double distance) {
		this.id = id;
		this.ls = ls;
		this.ss = ss;
		this.n = n;
		this.rep = rep;
		this.extent = extent;
		this.nnDist = nnDist;
		this.points = points;
		this.distance = distance;
	}

	public ClusterFeatureDataBubbles(LinkedList<ClusterFeatureDataBubbles> cf) {
		this.setCfs(cf);
	}

	public ClusterFeatureDataBubbles(int id, double[] ls, double[] ss, int n,
			double[] rep, double extent, double nnDist, DataPoints points,
			double distance, DataBubbles bubbles) {
		this.id = id;
		this.ls = ls;
		this.ss = ss;
		this.n = n;
		this.rep = rep;
		this.extent = extent;
		this.nnDist = nnDist;
		this.points = points;
		this.distance = distance;
		this.bubbles = bubbles;
	}

	public ClusterFeatureDataBubbles(String key, int id, double[] ls,
			double[] ss, int n, double[] rep, double extent, double nnDist,
			DataPoints dataPoints, double distance) {
		this.setKey(key);
		this.setId(id);
		this.setLs(ls);
		this.setSs(ss);
		this.setN(n);
		this.setRep(rep);
		this.setExtent(extent);
		this.setNnDist(nnDist);
		this.setPoints(dataPoints);
		this.setDistance(distance);
	}

	public ClusterFeatureDataBubbles(DataBubbles dataB) {
		this.setBubbles(dataB);
	}

	public double getDistance() {
		return this.distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public double[] getLs() {
		return this.ls;
	}

	public int getId() {
		return this.id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public void setLs(double[] ls) {
		this.ls = ls;
	}

	public double[] getSs() {
		return this.ss;
	}

	public void setSs(double[] ss) {
		this.ss = ss;
	}

	public int getN() {
		return this.n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public double[] getRep() {
		return this.rep;
	}

	public void setRep(double[] rep) {
		this.rep = rep;
	}

	public double getExtent() {
		return this.extent;
	}

	public void setExtent(double extent) {
		this.extent = extent;
	}

	public double getNnDist() {
		return this.nnDist;
	}

	public void setNnDist(double nnDist) {
		this.nnDist = nnDist;
	}

	public void setPoints(DataPoints points) {
		this.points = points;
	}

	public DataPoints getPoints() {
		return this.points;
	}

	public double[] calculateRep(double[] ls, int n, int col) {
		double[] _rep = new double[col];
		for (int i = 0; i < _rep.length; i++) {
			_rep[i] = ls[i] / n;
		}
		return _rep;
	}

	public double calculateExtent(double[] ls, double[] ss, int n, int col) {
		double sum = 0.0;
		for (int i = 0; i < ss.length; i++) {
			sum = sum
					+ (((2 * n * ss[i]) - (2 * (ls[i] * ls[i]))) / (n * (n - 1)));
		}
		return Math.sqrt(sum);
	}

	public double calculateNndist(double extent, int n, int col) {
		double _nnDist;
		double x = (double) 1 / n;
		double y = (double) 1 / col;
		_nnDist = (Math.pow(x, y) * extent);
		return _nnDist;
	}

	public LinkedList<ClusterFeatureDataBubbles> getCfs() {
		return this.cfs;
	}

	public void setCfs(LinkedList<ClusterFeatureDataBubbles> cfs) {
		this.cfs = cfs;
	}

	public DataBubbles getBubbles() {
		return bubbles;
	}

	public void setBubbles(DataBubbles bubbles) {
		this.bubbles = bubbles;
	}

	public String getKey() {
		return key;
	}

	public void setKey(String key) {
		this.key = key;
	}

	@Override
	public String toString() {
		return "ClusterFeatureDataBubbles [id=" + id + ", ls="
				+ Arrays.toString(ls) + ", ss=" + Arrays.toString(ss) + ", n="
				+ n + ", rep=" + Arrays.toString(rep) + ", extent=" + extent
				+ ", nnDist=" + nnDist + ", points=" + points + ", key=" + key
				+ "]";
	}
}
