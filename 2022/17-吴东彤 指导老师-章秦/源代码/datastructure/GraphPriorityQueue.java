package datastructure;

import java.io.Serializable;
import java.util.Comparator;

public class GraphPriorityQueue implements Comparator<CompleteGraph>, Serializable{
	
	private static final long serialVersionUID = 1L;

	public int compare(CompleteGraph g1, CompleteGraph g2) {
		if (g1.getWeight() < g2.getVertex2()) {
			return -1;
		}
		if (g1.getWeight() > g2.getWeight()) {
			return 1;
		}
		return 0;
	}
}
