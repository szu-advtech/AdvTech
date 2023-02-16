package partition.reducers;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;

import datastructure.ClusterFeatureDataBubbles;


public class UpdateBubblesReducer implements
		Function2<ClusterFeatureDataBubbles, ClusterFeatureDataBubbles, ClusterFeatureDataBubbles> {

	private static final long serialVersionUID = 1L;
	private Broadcast<double[][]> broadcastSampling;
	private double[] rep;
	private double extent;
	private double nnDist;
	
	public UpdateBubblesReducer(Broadcast<double[][]> samples){
		this.broadcastSampling = samples;
	}
	

	public ClusterFeatureDataBubbles call(ClusterFeatureDataBubbles cf1, ClusterFeatureDataBubbles cf2) throws Exception {
		for (int i = 0; i < this.broadcastSampling.getValue()[0].length; i++) {
			cf1.getLs()[i] = cf1.getLs()[i] + cf2.getLs()[i];
			cf1.getSs()[i] = cf1.getSs()[i] + cf2.getSs()[i];
		}
		
		ClusterFeatureDataBubbles bubbles = new ClusterFeatureDataBubbles();
		int nn = cf1.getN() + cf2.getN();

		this.rep = bubbles.calculateRep(cf1.getLs(), nn, cf1.getLs().length);
		this.extent = bubbles.calculateExtent(cf1.getLs(), cf1.getSs(), nn, cf1.getSs().length);
		this.nnDist = bubbles.calculateNndist(extent, nn, cf1.getLs().length);
		
		return new ClusterFeatureDataBubbles(cf1.getKey(), cf1.getLs(), cf1.getSs(), nn, this.rep, extent, nnDist);
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

}
