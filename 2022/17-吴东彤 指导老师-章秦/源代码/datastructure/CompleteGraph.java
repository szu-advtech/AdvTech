package datastructure;

import java.io.Serializable;
import java.util.LinkedList;

public class CompleteGraph implements Serializable{

	private static final long serialVersionUID = 1L;
	private int vertex1;
	private int vertex2;
	private int tmpId1;
	private int tmpId2;
	private double weight;
	private LinkedList<CompleteGraph> graph;

	public CompleteGraph() {
	}

	public CompleteGraph(int vertex1, int vertex2, int tmpId1, int tmpId2, double weight) {
         this.setVertex1(vertex1);
         this.setVertex2(vertex2);
         this.setTmpId1(tmpId1);
         this.setTmpId2(tmpId2);
         this.setWeight(weight);
	}
	
	public CompleteGraph(int vertex1, int vertex2, double weight) {
        this.setVertex1(vertex1);
        this.setVertex2(vertex2);
        this.setWeight(weight);
	}
	

	public CompleteGraph(LinkedList<CompleteGraph> graph) {
		this.setGraph(graph);
	}

	public int getVertex1() {
		return this.vertex1;
	}

	public void setVertex1(int vertex1) {
		this.vertex1 = vertex1;
	}

	public int getVertex2() {
		return this.vertex2;
	}

	public void setVertex2(int vertex2) {
		this.vertex2 = vertex2;
	}

	public double getWeight() {
		return this.weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public int getTmpId1() {
		return tmpId1;
	}

	public void setTmpId1(int tmpId1) {
		this.tmpId1 = tmpId1;
	}

	public int getTmpId2() {
		return tmpId2;
	}

	public void setTmpId2(int tmpId2) {
		this.tmpId2 = tmpId2;
	}

	public LinkedList<CompleteGraph> getGraph() {
		return graph;
	}

	public void setGraph(LinkedList<CompleteGraph> graph) {
		this.graph = graph;
	}

	@Override
	public String toString() {
		return "CompleteGraph [vertex1=" + vertex1 + ", vertex2=" + vertex2
				+ ", tmpId1=" + tmpId1 + ", tmpId2=" + tmpId2 + ", weight="
				+ weight + "]\n";
	}
}
