package datastructure;

import java.io.Serializable;
import java.util.Arrays;

public class ClustersBubbles implements Serializable{
     
	private static final long serialVersionUID = 1L;
	private int[][] indicesMatch;
	private int[] indicesClusters;
    private int[] indicesBubbles;
    private int numberClusters;
    
     
     public ClustersBubbles(){
    	 
     }
     public ClustersBubbles(int[][] indicesMatch, int[] indicesClusters, int[] indicesBubbles, int numClusters){
    	 this.setIndicesMatch(indicesMatch);
    	 this.setIndicesClusters(indicesClusters);
    	 this.setIndicesBubbles(indicesBubbles);
    	 this.setNumberClusters(numClusters);
     }
	public int[] getIndicesClusters() {
		return indicesClusters;
	}
	public void setIndicesClusters(int[] indicesClusters) {
		this.indicesClusters = indicesClusters;
	}
	public int[] getIndicesBubbles() {
		return indicesBubbles;
	}
	public void setIndicesBubbles(int[] indicesBubbles) {
		this.indicesBubbles = indicesBubbles;
	}
	@Override
	public String toString() {
		return "ClustersBubbles [indicesClusters="
				+ Arrays.toString(indicesClusters) + ", indicesBubbles="
				+ Arrays.toString(indicesBubbles) + "]";
	}
	public int[][] getIndicesMatch() {
		return indicesMatch;
	}
	public void setIndicesMatch(int[][] indicesMatch) {
		this.indicesMatch = indicesMatch;
	}
	public int getNumberClusters() {
		return numberClusters;
	}
	public void setNumberClusters(int numberClusters) {
		this.numberClusters = numberClusters;
	}
}
