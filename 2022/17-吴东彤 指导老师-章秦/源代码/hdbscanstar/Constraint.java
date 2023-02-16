package hdbscanstar;

/**
 * A clustering constraint (either a must-link or cannot-link constraint between two points).
 * @author zjullion
 */
public class Constraint {

	// ------------------------------ PRIVATE VARIABLES ------------------------------
	
	private CONSTRAINT_TYPE type;
	private int pointA;
	private int pointB;

	// ------------------------------ CONSTANTS ------------------------------
	
	public static final String MUST_LINK_TAG = "ml";
	public static final String CANNOT_LINK_TAG = "cl";
	
	public static enum CONSTRAINT_TYPE {
		MUST_LINK,
		CANNOT_LINK
	}

	// ------------------------------ CONSTRUCTORS ------------------------------
	
	/**
	 * Creates a new constraint.
	 * @param pointA The first point involved in the constraint
	 * @param pointB The second point involved in the constraint
	 * @param type The CONSTRAINT_TYPE of the constraint
	 */
	public Constraint(int pointA, int pointB, CONSTRAINT_TYPE type) {
		this.pointA = pointA;
		this.pointB = pointB;
		this.type = type;
	}

	// ------------------------------ PUBLIC METHODS ------------------------------

	// ------------------------------ PRIVATE METHODS ------------------------------

	// ------------------------------ GETTERS & SETTERS ------------------------------
	
	public int getPointA() {
		return this.pointA;
	}
	
	public int getPointB() {
		return this.pointB;
	}
	
	public CONSTRAINT_TYPE getType() {
		return this.type;
	}
}