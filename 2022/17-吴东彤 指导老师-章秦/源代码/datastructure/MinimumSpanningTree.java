package datastructure;

import java.io.Serializable;
import java.util.Comparator;
import java.util.LinkedList;

public class MinimumSpanningTree implements Comparator<MinimumSpanningTree>, Serializable{

	private static final long serialVersionUID = 1L;
	private int vertice1;
	private int vertice2;
	private int fake1;
	private int fake2;
	private double weight;
	private int node;
	private LinkedList<MinimumSpanningTree> tree;
	private MinimumSpanningTree[] vectorMst;
	
	public MinimumSpanningTree() {

	}

	public MinimumSpanningTree(int vertice1, int vertice2, double weight, int fakeId1, int fakeId2, int node) {
		this.setVertice1(vertice1);
		this.setVertice2(vertice2);
		this.setWeight(weight);
		this.setFake1(fakeId1);
		this.setFake2(fakeId2);
		this.setNode(node);
	}
	
	public MinimumSpanningTree(LinkedList<MinimumSpanningTree> tree){
		this.setTree(tree);
	}
	
	public MinimumSpanningTree(MinimumSpanningTree[] mst){
		this.setVectorMst(mst);
	}
	
	
	public int compare(MinimumSpanningTree v1, MinimumSpanningTree v2) {
		if(v1.getWeight() > v2.getWeight()){
			return 1;
		}
		else if(v1.getWeight() < v2.getWeight()){
			return -1;
		}
		return 0;
	}
	
	

	public int getVertice1() {
		return vertice1;
	}

	public void setVertice1(int vertice1) {
		this.vertice1 = vertice1;
	}

	public int getVertice2() {
		return vertice2;
	}

	public void setVertice2(int vertice2) {
		this.vertice2 = vertice2;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public void quicksortByEdgeWeight() {

	}

	public LinkedList<MinimumSpanningTree> getTree() {
		return tree;
	}

	public void setTree(LinkedList<MinimumSpanningTree> tree) {
		this.tree = tree;
	}

	public int getFake1() {
		return fake1;
	}

	public void setFake1(int fake1) {
		this.fake1 = fake1;
	}

	public int getFake2() {
		return fake2;
	}

	public void setFake2(int fake2) {
		this.fake2 = fake2;
	}

	public int getNode() {
		return node;
	}

	public void setNode(int node) {
		this.node = node;
	}

	public MinimumSpanningTree[] getVectorMst() {
		return vectorMst;
	}

	public void setVectorMst(MinimumSpanningTree[] vectorMst) {
		this.vectorMst = vectorMst;
	}
}
