package datastructure;

import java.io.Serializable;
import java.util.*;

public class SortGraph implements Comparator<CompleteGraph>, Serializable {

	private static final long serialVersionUID = 1L;

	public int compare(CompleteGraph v1, CompleteGraph v2) {
		if(v1.getWeight() > v2.getWeight()){
			return 1;
		}
		else if(v1.getWeight() < v2.getWeight()){
			return -1;
		}
		return 0;
	}
}
