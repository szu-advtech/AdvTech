package distance;

import java.io.Serializable;

/**
 * Computes cosine similarity between two points, d = 1 - ((X�Y) / (||X||*||Y||))
 * @author zjullion
 */
public class CosineSimilarity implements DistanceCalculator, Serializable {

	// ------------------------------ PRIVATE VARIABLES ------------------------------

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	public CosineSimilarity() {
	}

	// ------------------------------ PUBLIC METHODS ------------------------------
	
	public double computeDistance(double[] attributesOne, double[] attributesTwo) {
		double dotProduct = 0;
		double magnitudeOne = 0;
		double magnitudeTwo = 0;
		
		for (int i = 0; i < attributesOne.length && i < attributesTwo.length; i++) {
			dotProduct+= (attributesOne[i] * attributesTwo[i]);
			magnitudeOne+= (attributesOne[i] * attributesOne[i]);
			magnitudeTwo+= (attributesTwo[i] * attributesTwo[i]);
		}
		
		return 1 - (dotProduct / Math.sqrt(magnitudeOne * magnitudeTwo));
	}
	
	
	public String getName() {
		return "cosine";
	}

	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------

}