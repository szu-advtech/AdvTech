package distance;

import java.io.Serializable;

/**
 * Computes the supremum distance between two points, d = max[(x1-y1), (x2-y2), ... ,(xn-yn)].
 * @author zjullion
 */
public class SupremumDistance implements DistanceCalculator, Serializable {

	// ------------------------------ PRIVATE VARIABLES ------------------------------

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	public SupremumDistance() {
	}

	// ------------------------------ PUBLIC METHODS ------------------------------
	
	public double computeDistance(double[] attributesOne, double[] attributesTwo) {
		double distance = 0;
		
		for (int i = 0; i < attributesOne.length && i < attributesTwo.length; i++) {
			double difference = Math.abs(attributesOne[i] - attributesTwo[i]);
			if (difference > distance)
				distance = difference;
		}
		
		return distance;
	}
	
	
	public String getName() {
		return "supremum";
	}

	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------

}