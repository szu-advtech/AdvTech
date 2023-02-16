package datastructure;

import hdbscanstar.UndirectedGraph;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;

public class DataPoints implements Serializable {

	private static final long serialVersionUID = 1L;
	private int idPoint;
	private int tmpIdPoint;
	private int idBubble;
	private int labelCluster;
	private double[] point;
	private String key;
	private LinkedList<DataPoints> dataPoints;
	private Features features;
	private int[][] indices;
	private DataBubbles bubbles;
	private ClustersBubbles clusters;
	private UndirectedGraph mst;
	private LinkedList<ClusterFeatureDataBubbles> clusterFeatures;
	private String node;
	private MinimumSpanningTree localMst;

	public DataPoints() {
	}

	public DataPoints(int idPoint, int idBubble, int labelCluster, double[] point) {
		this.setIdPoint(idPoint);
		this.setIdBubble(idBubble);
		this.setLabelCluster(labelCluster);
		this.setPoint(point);
	}
	
	public DataPoints(int idPoint, int idBubble, String key, double[] point) {
		this.setIdPoint(idPoint);
		this.setIdBubble(idBubble);
		this.setKey(key);
		this.setPoint(point);
	}
	
	public DataPoints(int idPoint, int tmpIdPoint, int idBubble, int labelCluster, double[] point) {
		this.setIdPoint(idPoint);
		this.setTmpIdPoint(tmpIdPoint);
		this.setIdBubble(idBubble);
		this.setLabelCluster(labelCluster);
		this.setPoint(point);
	}
	
	public DataPoints(int idPoint, int tmpIdPoint, int idBubble, int labelCluster, String key, double[] point) {
		this.setIdPoint(idPoint);
		this.setTmpIdPoint(tmpIdPoint);
		this.setIdBubble(idBubble);
		this.setLabelCluster(labelCluster);
		this.setPoint(point);
		this.setKey(key);
	}
	
		
	public DataPoints(int idPoint, int tmpIdPoint, int idBubble, int labelCluster, double[] point, LinkedList<DataPoints> data) {
		this.setIdPoint(idPoint);
		this.setTmpIdPoint(tmpIdPoint);
		this.setIdBubble(idBubble);
		this.setLabelCluster(labelCluster);
		this.setPoint(point);
		this.setDataPoints(data);
	}
	
	public DataPoints(LinkedList<DataPoints> data) {
		this.setDataPoints(data);
	}
	
	public DataPoints(LinkedList<ClusterFeatureDataBubbles> cfs, boolean type){
		this.setClusterFeatures(cfs);
	}
	
	public DataPoints(DataBubbles bubbles){
	    this.setBubbles(bubbles);
	}
	
	public DataPoints(Features features){
		this.setFeatures(features);
	}
	
	public DataPoints(UndirectedGraph mst) {
		this.setMst(mst);
	}
	
	public DataPoints(int[][] indices) {
		this.setIndices(indices);
	}
	
	public DataPoints(ClustersBubbles clustersBubbles, LinkedList<DataPoints> dataPoints) {
		this.setClusters(clustersBubbles);
		this.setDataPoints(dataPoints);
	}

	public DataPoints(int idPoint, int tmpIdPoint, int idBubble, String key,
			double[] point, LinkedList<DataPoints> data) {
		this.setIdPoint(idPoint);
		this.setTmpIdPoint(tmpIdPoint);
		this.setIdBubble(idBubble);
		this.setKey(key);
		this.setPoint(point);
	}

	public DataPoints(int currentIdPoint, int indexBubble, int labelCluster, String key,
			double[] currentPoint) {
		this.setIdPoint(currentIdPoint);
		this.setIdBubble(indexBubble);
		this.setLabelCluster(labelCluster);
		this.setKey(key);
		this.setPoint(currentPoint);
	}

	public DataPoints(LinkedList<ClusterFeatureDataBubbles> features, String keyCurrentPoint, boolean b) {
		this.setClusterFeatures(features);
		this.setKey(keyCurrentPoint);
	}

	public DataPoints(UndirectedGraph mst, String node) {
		this.setMst(mst);
	}

	public DataPoints(MinimumSpanningTree mst) {
		this.setLocalMst(mst);
	}

	public int getIdPoint() {
		return idPoint;
	}

	public void setIdPoint(int idPoint) {
		this.idPoint = idPoint;
	}

	public int getLabelCluster() {
		return this.labelCluster;
	}

	public void setLabelCluster(int label) {
		this.labelCluster = label;
	}

	public int getIdBubble() {
		return idBubble;
	}

	public void setIdBubble(int idBubble) {
		this.idBubble = idBubble;
	}

	public double[] getPoint() {
		return point;
	}

	public void setPoint(double[] point) {
		this.point = point;
	}

	public int getTmpIdPoint() {
		return tmpIdPoint;
	}

	public void setTmpIdPoint(int tmpIdPoint) {
		this.tmpIdPoint = tmpIdPoint;
	}

	public LinkedList<DataPoints> getDataPoints() {
		return dataPoints;
	}

	public void setDataPoints(LinkedList<DataPoints> dataPoints) {
		this.dataPoints = dataPoints;
	}

	public UndirectedGraph getMst() {
		return mst;
	}

	public void setMst(UndirectedGraph mst) {
		this.mst = mst;
	}

	public DataBubbles getBubbles() {
		return bubbles;
	}

	public void setBubbles(DataBubbles bubbles) {
		this.bubbles = bubbles;
	}

	public Features getFeatures() {
		return features;
	}

	public void setFeatures(Features features) {
		this.features = features;
	}

	public int[][] getIndices() {
		return indices;
	}

	public void setIndices(int[][] indices) {
		this.indices = indices;
	}

	public ClustersBubbles getClusters() {
		return clusters;
	}

	public void setClusters(ClustersBubbles clusters) {
		this.clusters = clusters;
	}

	public String getKey() {
		return key;
	}

	public void setKey(String key) {
		this.key = key;
	}

	public LinkedList<ClusterFeatureDataBubbles> getClusterFeatures() {
		return clusterFeatures;
	}

	public void setClusterFeatures(LinkedList<ClusterFeatureDataBubbles> clusterFeatures) {
		this.clusterFeatures = clusterFeatures;
	}

	public String getNode() {
		return node;
	}

	public void setNode(String node) {
		this.node = node;
	}

	public MinimumSpanningTree getLocalMst() {
		return localMst;
	}

	public void setLocalMst(MinimumSpanningTree localMst) {
		this.localMst = localMst;
	}

	@Override
	public String toString() {
		return "DataPoints [localMst=" + localMst + "]";
	}
}
