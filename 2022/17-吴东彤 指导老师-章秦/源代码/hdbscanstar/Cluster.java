package hdbscanstar;

import java.util.ArrayList;
import java.util.TreeSet;


/**
 * An HDBSCAN* cluster, which will have a birth level, death level, stability, and constraint 
 * satisfaction once fully constructed.
 * @author zjullion
 */
public class Cluster {

	// ------------------------------ PRIVATE VARIABLES ------------------------------
	
	private int label;
	private double birthLevel;
	private double deathLevel;
	private int numPoints;
	private long fileOffset;	//First level where points with this cluster's label appear
	
	private double stability;
	private double propagatedStability;
	
	private double propagatedLowestChildDeathLevel;
	
	private int numConstraintsSatisfied;
	private int propagatedNumConstraintsSatisfied;
	private TreeSet<Integer> virtualChildCluster;

	private Cluster parent;
	private boolean hasChildren;
	public ArrayList<Cluster> propagatedDescendants;
	

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * Creates a new Cluster.
	 * @param label The cluster label, which should be globally unique
	 * @param parent The cluster which split to create this cluster
	 * @param birthLevel The MST edge level at which this cluster first appeared
	 * @param numPoints The initial number of points in this cluster
	 */
	public Cluster(int label, Cluster parent, double birthLevel, int numPoints) {
		this.label = label;
		this.birthLevel = birthLevel;
		this.deathLevel = 0;
		this.numPoints = numPoints;
		this.fileOffset = 0;
		
		this.stability = 0;
		this.propagatedStability = 0;
		
		this.propagatedLowestChildDeathLevel = Double.MAX_VALUE;
		
		this.numConstraintsSatisfied = 0;
		this.propagatedNumConstraintsSatisfied = 0;
		this.virtualChildCluster = new TreeSet<Integer>();
		
		this.parent = parent;
		if (this.parent != null)
			this.parent.hasChildren = true;
		this.hasChildren = false;
		this.propagatedDescendants = new ArrayList<Cluster>(1);
	}
	

	// ------------------------------ PUBLIC METHODS ------------------------------
	
	/**
	 * Removes the specified number of points from this cluster at the given edge level, which will
	 * update the stability of this cluster and potentially cause cluster death.  If cluster death
	 * occurs, the number of constraints satisfied by the virtual child cluster will also be calculated.
	 * @param numPoints The number of points to remove from the cluster
	 * @param level The MST edge level at which to remove these points
	 */
	public void detachPoints(int numPoints, int countMembers, double level) {
		this.numPoints-=numPoints;
		this.stability+=((numPoints + countMembers) * (1/level - 1/this.birthLevel));
		
		if (this.numPoints == 0)
			this.deathLevel = level;
		else if (this.numPoints < 0)
			throw new IllegalStateException("Cluster cannot have less than 0 points.");
	}
	

	/**
	 * This cluster will propagate itself to its parent if its number of satisfied constraints is
	 * higher than the number of propagated constraints.  Otherwise, this cluster propagates its
	 * propagated descendants.  In the case of ties, stability is examined.
	 * Additionally, this cluster propagates the lowest death level of any of its descendants to its
	 * parent.
	 */
	public void propagate() {
		if (this.parent != null) {
			
			//Propagate lowest death level of any descendants:
			if (this.propagatedLowestChildDeathLevel == Double.MAX_VALUE)
				this.propagatedLowestChildDeathLevel = this.deathLevel;
			if (this.propagatedLowestChildDeathLevel < this.parent.propagatedLowestChildDeathLevel)
				this.parent.propagatedLowestChildDeathLevel = this.propagatedLowestChildDeathLevel;
			
			//If this cluster has no children, it must propagate itself:
			if (!this.hasChildren) {
				this.parent.propagatedNumConstraintsSatisfied+= this.numConstraintsSatisfied;
				this.parent.propagatedStability+= this.stability;
				this.parent.propagatedDescendants.add(this);
			}
			
			else if (this.numConstraintsSatisfied > this.propagatedNumConstraintsSatisfied) {
				this.parent.propagatedNumConstraintsSatisfied+= this.numConstraintsSatisfied;
				this.parent.propagatedStability+= this.stability;
				this.parent.propagatedDescendants.add(this);
			}
			
			else if (this.numConstraintsSatisfied < this.propagatedNumConstraintsSatisfied) {
				this.parent.propagatedNumConstraintsSatisfied+= this.propagatedNumConstraintsSatisfied;
				this.parent.propagatedStability+= this.propagatedStability;
				this.parent.propagatedDescendants.addAll(this.propagatedDescendants);
			}
			
			else if (this.numConstraintsSatisfied == this.propagatedNumConstraintsSatisfied) {
				
				//Chose the parent over descendants if there is a tie in stability:
				if (this.stability >= this.propagatedStability) {
					this.parent.propagatedNumConstraintsSatisfied+= this.numConstraintsSatisfied;
					this.parent.propagatedStability+= this.stability;
					this.parent.propagatedDescendants.add(this);
				}
					
				else {
					this.parent.propagatedNumConstraintsSatisfied+= this.propagatedNumConstraintsSatisfied;
					this.parent.propagatedStability+= this.propagatedStability;
					this.parent.propagatedDescendants.addAll(this.propagatedDescendants);
				}	
			}	
		}
	}
	
	
	public void addPointsToVirtualChildCluster(TreeSet<Integer> points) {
		this.virtualChildCluster.addAll(points);
	}
	
	
	public boolean virtualChildClusterContaintsPoint(int point) {
		return this.virtualChildCluster.contains(point);
	}
	
	
	public void addVirtualChildConstraintsSatisfied(int numConstraints) {
		this.propagatedNumConstraintsSatisfied+= numConstraints;
	}
	
	
	public void addConstraintsSatisfied(int numConstraints) {
		this.numConstraintsSatisfied+= numConstraints;
	}
	
	
	/**
	 * Sets the virtual child cluster to null, thereby saving memory.  Only call this method after computing the
	 * number of constraints satisfied by the virtual child cluster.
	 */
	public void releaseVirtualChildCluster() {
		this.virtualChildCluster = null;
	}
	
	
	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------
	
	public int getLabel() {
		return this.label;
	}
	
	public Cluster getParent() {
		return this.parent;
	}
	
	public double getBirthLevel() {
		return this.birthLevel;
	}
	
	public double getDeathLevel() {
		return this.deathLevel;
	}
	
	public long getFileOffset() {
		return this.fileOffset;
	}
	
	public void setFileOffset(long offset) {
		this.fileOffset = offset;
	}

	public double getStability() {
		return this.stability;
	}
	
	public double getPropagatedLowestChildDeathLevel() {
		return this.propagatedLowestChildDeathLevel;
	}
	
	public int getNumConstraintsSatisfied() {
		return this.numConstraintsSatisfied;
	}
	
	public int getPropagatedNumConstraintsSatisfied() {
		return this.propagatedNumConstraintsSatisfied;
	}
	
	public ArrayList<Cluster> getPropagatedDescendants() {
		return this.propagatedDescendants;
	}
	
	public boolean hasChildren() {
		return this.hasChildren;
	}
}