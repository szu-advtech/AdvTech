package hdbscanstar;

import java.util.ArrayList;
import java.util.TreeSet;

/**
 * Creates a new Cluster.
 * @param label The cluster label, which should be globally unique
 * @param parent The cluster which split to create this cluster
 * @param birthLevel The MST edge level at which this cluster first appeared
 * @param numPoints The initial number of points in this cluster
 */

public class Clusters implements Comparable<Clusters>{
	private int label;
	private double birthLevel;
	private double deathLevel;
	private int numPoints;
	private TreeSet<Integer> members;
	private double stability;
	
	// private double propagatedLowestChildDeathLevel;
	private int parentId;
	private boolean hasChildren;
	private int numOfChildren;
	
	public Clusters(int label, int parent, double birthLevel, int numPoints, TreeSet<Integer> members) {
		this.label = label;
		this.birthLevel = birthLevel;
		this.deathLevel = Double.MAX_VALUE;
		this.numPoints = numPoints;
		this.members = members;
		this.stability = 0;
		this.parentId = parent;
		this.hasChildren = false;
		this.numOfChildren = 0;
	}
	
	public void detachPoints(int numPoints, int countMembers, double level) {
		this.numPoints-=numPoints;
		this.stability+=((numPoints + countMembers) * (1/level - 1/this.birthLevel));
		
		if (this.numPoints == 0)
			this.deathLevel = level;
		else if (this.numPoints < 0) // ***
			throw new IllegalStateException("Cluster cannot have less than 0 points.");
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public double getBirthLevel() {
		return birthLevel;
	}

	public void setBirthLevel(double birthLevel) {
		this.birthLevel = birthLevel;
	}

	public double getDeathLevel() {
		return deathLevel;
	}

	public void setDeathLevel(double deathLevel) {
		this.deathLevel = deathLevel;
	}

	public int getNumPoints() {
		return numPoints;
	}

	public void setNumPoints(int numPoints) {
		this.numPoints = numPoints;
	}

	public TreeSet<Integer> getMembers() {
		return members;
	}

	public void setMembers(TreeSet<Integer> members) {
		this.members = members;
	}

	public double getStability() {
		return stability;
	}

	public void setStability(double stability) {
		this.stability = stability;
	}

	public int getParentId() {
		return parentId;
	}

	public void setParentId(int parentId) {
		this.parentId = parentId;
	}

	public boolean isHasChildren() {
		return hasChildren;
	}

	public void setHasChildren(boolean hasChildren) {
		this.hasChildren = hasChildren;
	}

	public int getNumOfChildren() {
		return numOfChildren;
	}

	public void setNumOfChildren(int numOfChildren) {
		this.numOfChildren = numOfChildren;
	}

	public int compareTo(Clusters o) {
		return this.label - o.getLabel();
	}
}
