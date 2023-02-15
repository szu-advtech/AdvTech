package reducers;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;

import datastructure.ClusterFeatureDataBubbles;
import distance.DistanceCalculator;

public class ConstructDataBubblesReducer
		implements
		Function2<ClusterFeatureDataBubbles, ClusterFeatureDataBubbles, ClusterFeatureDataBubbles> {
	/** 
	 */
	private static final long serialVersionUID = 1L;
	private Broadcast<double[][]> broadcastSampling;
	private double[] rep;
	private double extent;
	private double nnDist;
	// ------------------------------ CONSTANTS ------------------------------
	public static final String WARNING_MESSAGE = "----------------------------------------------- WARNING -----------------------------------------------\n"
			+ "With your current settings, the K-NN density estimate is discontinuous as it is not well-defined\n"
			+ "(infinite) for some data objects, either due to replicates in the data (not a set) or due to numerical\n"
			+ "roundings. This does not affect the construction of the density-based clustering hierarchy, but\n"
			+ "it affects the computation of cluster stability by means of relative excess of mass. For this reason,\n"
			+ "the post-processing routine to extract a flat partition containing the most stable clusters may\n"
			+ "produce unexpected results. It may be advisable to increase the value of MinPts and/or M_clSize.\n"
			+ "-------------------------------------------------------------------------------------------------------";

	
    public ConstructDataBubblesReducer(){
    	
    }
	
	public ConstructDataBubblesReducer(Broadcast<double[][]> b,
			DistanceCalculator dist) {
		this.broadcastSampling = b;
		rep = new double[this.broadcastSampling.getValue()[0].length];
		extent = 0;
		nnDist = 0;
	}

	public Broadcast<double[][]> getBroadcastSampling() {
		return broadcastSampling;
	}

	public void setBroadcastSampling(Broadcast<double[][]> broadcastSampling) {
		this.broadcastSampling = broadcastSampling;
	}

	public double[] getRep() {
		return rep;
	}

	public void setRep(double[] rep) {
		this.rep = rep;
	}

	public double getExtent() {
		return extent;
	}

	public void setExtent(double extent) {
		this.extent = extent;
	}

	public double getNnDist() {
		return nnDist;
	}

	public void setNnDist(double nnDist) {
		this.nnDist = nnDist;
	}

	public ClusterFeatureDataBubbles call(ClusterFeatureDataBubbles cf1,
			ClusterFeatureDataBubbles cf2) throws Exception {

		for (int i = 0; i < this.broadcastSampling.getValue()[0].length; i++) {
			cf1.getLs()[i] = cf1.getLs()[i] + cf2.getLs()[i];
			cf1.getSs()[i] = cf1.getSs()[i] + cf2.getSs()[i];
		}
		
		ClusterFeatureDataBubbles bubbles = new ClusterFeatureDataBubbles();
		int nn = cf1.getN() + cf2.getN();

		this.rep = bubbles.calculateRep(cf1.getLs(), nn, cf1.getLs().length);
		this.extent = bubbles.calculateExtent(cf1.getLs(), cf1.getSs(), nn, cf1.getSs().length);
		this.nnDist = bubbles.calculateNndist(extent, nn, cf1.getLs().length);
		
	  return new ClusterFeatureDataBubbles(cf1.getId(), cf1.getLs(), cf1.getSs(), nn, this.rep, extent, nnDist);
	}
}
