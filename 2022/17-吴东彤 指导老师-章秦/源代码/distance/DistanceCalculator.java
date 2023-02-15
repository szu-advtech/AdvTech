package distance;

/**
 * An interface for classes which compute the distance between two points (where points are
 * represented as arrays of doubles).
 * @author zjullion
 */
public interface DistanceCalculator {

	/**
	 * Computes the distance between two points.  Note that larger values indicate that the two points
	 * are farther apart.
	 * @param attributesOne The attributes of the first point
	 * @param attributesTwo The attributes of the second point
	 * @return A double for the distance between the two points
	 */
	public double computeDistance(double[] attributesOne, double[] attributesTwo);
	
	
	public String getName();
}
