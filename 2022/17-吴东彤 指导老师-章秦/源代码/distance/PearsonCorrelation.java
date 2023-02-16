package distance;

import java.io.Serializable;

/**
 * Computes the euclidean distance between two points, d = 1 - (cov(X,Y) / (std_dev(X) * std_dev(Y)))
 * @author zjullion
 */
public class PearsonCorrelation implements DistanceCalculator, Serializable {

	// ------------------------------ PRIVATE VARIABLES ------------------------------

	// ------------------------------ CONSTANTS ------------------------------

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	public PearsonCorrelation() {
	}

	// ------------------------------ PUBLIC METHODS ------------------------------
	
	public double computeDistance(double[] attributesOne, double[] attributesTwo) {
		double meanOne = 0;
		double meanTwo = 0;
		
		for (int i = 0; i < attributesOne.length && i < attributesTwo.length; i++) {
			meanOne+= attributesOne[i];
			meanTwo+= attributesTwo[i];
		}
		
		meanOne = meanOne / attributesOne.length;
		meanTwo = meanTwo / attributesTwo.length;
		
		double covariance = 0;
		double standardDeviationOne = 0;
		double standardDeviationTwo = 0;
		
		for (int i = 0; i < attributesOne.length && i < attributesTwo.length; i++) {
			covariance+= ((attributesOne[i] - meanOne) * (attributesTwo[i] - meanTwo));
			standardDeviationOne+= ((attributesOne[i] - meanOne) * (attributesOne[i] - meanOne));
			standardDeviationTwo+= ((attributesTwo[i] - meanTwo) * (attributesTwo[i] - meanTwo));
		}
		
		return (1 - (covariance / Math.sqrt(standardDeviationOne * standardDeviationTwo)));
	}
	
	
	public String getName() {
		return "pearson";
	}

	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------

}
