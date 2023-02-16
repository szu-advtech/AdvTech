package hdbscanstar;

/**
 * Simple storage class that keeps the outlier score, core distance, and id (index) for a single point.
 * OutlierScores are sorted in ascending order by outlier score, with core distances used to break
 * outlier score ties, and ids used to break core distance ties.
 * @author zjullion
 */
public class OutlierScore implements Comparable<OutlierScore> {

	// ------------------------------ PRIVATE VARIABLES ------------------------------
	
	private double score;
	private double coreDistance;
	private int id;

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * Creates a new OutlierScore for a given point.
	 * @param score The outlier score of the point
	 * @param coreDistance The point's core distance
	 * @param id The id (index) of the point
	 */
	public OutlierScore(double score, double coreDistance, int id) {
		this.score = score;
		this.coreDistance = coreDistance;
		this.id = id;
	}


	// ------------------------------ PUBLIC METHODS ------------------------------
	
	public int compareTo(OutlierScore other) {
		if (this.score > other.score)
			return 1;
		else if (this.score < other.score)
			return -1;
		else {
			
			if (this.coreDistance > other.coreDistance)
				return 1;
			else if (this.coreDistance < other.coreDistance)
				return -1;
			else
				return this.id-other.id;
		}
	}

	
	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------
	
	public double getScore() {
		return this.score;
	}
	
	public int getId() {
		return this.id;
	}
}