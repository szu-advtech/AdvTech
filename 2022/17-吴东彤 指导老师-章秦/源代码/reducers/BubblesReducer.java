package reducers;

import java.util.LinkedList;

import org.apache.spark.api.java.function.Function2;

import datastructure.CoreDistances;
import datastructure.MinimumSpanningTree;

public class BubblesReducer implements
		Function2<CoreDistances, CoreDistances, CoreDistances> {

	private static final long serialVersionUID = 1L;

	public CoreDistances call(CoreDistances v1, CoreDistances v2)
			throws Exception {
//		LinkedList<MinimumSpanningTree> mst = new LinkedList<MinimumSpanningTree>();
//		if(v1.getMst().getFirst().getWeight() > v2.getMst().getFirst().getWeight()){
//		   mst.add(v2.getMst().getFirst());	
//		}
//		else{
//		   mst.add(v1.getMst().getFirst());	
//		}
		v1.getMst().addAll(v2.getMst());
//		SortMST sort = new SortMST();
//		Collections.sort(v1.getMst(), sort);
//		LinkedList<MinimumSpanningTree> mst = new LinkedList<MinimumSpanningTree>();
//		mst.add(v1.getMst().remove(1));
		return new CoreDistances(null, v1.getMst());
	}

}
