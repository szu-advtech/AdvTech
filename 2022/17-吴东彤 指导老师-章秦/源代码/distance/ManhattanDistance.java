package distance;

import java.io.Serializable;

/**
 * Computes the manhattan distance between two points, d = |x1-y1| + |x2-y2| + ... + |xn-yn|.
 * @author zjullion
 */
public class ManhattanDistance implements DistanceCalculator, Serializable {

	// ------------------------------ PRIVATE VARIABLES ------------------------------

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	public ManhattanDistance() {
	}

	// ------------------------------ PUBLIC METHODS ------------------------------
	
	public double computeDistance(double[] attributesOne, double[] attributesTwo) {
		double distance = 0;
		
		for (int i = 0; i < attributesOne.length && i < attributesTwo.length; i++) {
			distance+= Math.abs(attributesOne[i] - attributesTwo[i]);
		}
		
		return distance;
	}
	
	
	public String getName() {
		return "manhattan";
	}

	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------

}
