package mappers;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;
import datastructure.CoreDistances;
import datastructure.MinimumSpanningTree;
import datastructure.UF;

public class InterEdgesMapper implements
		PairFunction<Tuple2<String, CoreDistances>, String, CoreDistances> {

	private static final long serialVersionUID = 1L;
	private static int[] id;
	private int tam;
	private int overallWeight;
	private UF uf;
	
	public InterEdgesMapper(int tam){
		this.uf = new UF(tam);
		this.overallWeight = 0;
	}

	public Tuple2<String, CoreDistances> call(Tuple2<String, CoreDistances> t)
			throws Exception {
		String[] values = t._1().split(" ");
		int[] sArray = new int[values.length];
		
		for(int i = 0; i < sArray.length; i++){
		  sArray[i] = Integer.parseInt(values[i]);
		}
		if (this.uf.find(sArray[0]) != this.uf.find(sArray[1])) {
			this.uf.union(sArray[0], sArray[1]);
			//mst.add(new MinimumSpanningTree(t._2.getGraph().get(i).getVertex1(), t._2.getGraph().get(i).getVertex2(), t._2.getGraph().get(i).getWeight(), overallWeight));
		} else{
		   //t._2.getGraph().remove(i);	
		}
		
		return null;
	}

	public static int[] getId() {
		return id;
	}

	public static void setId(int[] id) {
		InterEdgesMapper.id = id;
	}

	public UF getUf() {
		return uf;
	}

	public void setUf(UF uf) {
		this.uf = uf;
	}

}
