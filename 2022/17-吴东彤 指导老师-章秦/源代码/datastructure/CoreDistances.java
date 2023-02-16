package datastructure;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;

public class CoreDistances implements Serializable {

	private static final long serialVersionUID = 1L;
	private int idPoint;
	private int tmpIdPoint;
	private double coreDistance;
	private double[] point;
	private MinimumSpanningTree _mst;
	private LinkedList<CoreDistances> list;
	private LinkedList<CompleteGraph> graph;
	private LinkedList<MinimumSpanningTree> mst;

	public CoreDistances(int idPoints, int tmpIdPoints, double coreDistance, double[] point) {
		this.setIdPoint(idPoints);
		this.setTmpIdPoint(tmpIdPoints);
		this.setCoreDistance(coreDistance);
		this.setPoint(point);
	}

	public CoreDistances(LinkedList<CompleteGraph> graph,
			LinkedList<MinimumSpanningTree> mst) {
		this.setGraph(graph);
		this.setMst(mst);
	}

	public CoreDistances() {
	}
	
	public CoreDistances(MinimumSpanningTree _mst) {
		this.set_mst(_mst);
	}

	public int getIdPoint() {
		return idPoint;
	}

	public void setIdPoint(int idPoint) {
		this.idPoint = idPoint;
	}

	public double getCoreDistance() {
		return coreDistance;
	}

	public void setCoreDistance(double coreDistance) {
		this.coreDistance = coreDistance;
	}

	public double[] getPoint() {
		return point;
	}

	public void setPoint(double[] point) {
		this.point = point;
	}

	public LinkedList<CoreDistances> getList() {
		return list;
	}

	public void setList(LinkedList<CoreDistances> list) {
		this.list = list;
	}

	public LinkedList<CompleteGraph> getGraph() {
		return graph;
	}

	public void setGraph(LinkedList<CompleteGraph> graph) {
		this.graph = graph;
	}

	public int getTmpIdPoint() {
		return tmpIdPoint;
	}

	public void setTmpIdPoint(int tmpIdPoint) {
		this.tmpIdPoint = tmpIdPoint;
	}

	public LinkedList<MinimumSpanningTree> getMst() {
		return mst;
	}

	public void setMst(LinkedList<MinimumSpanningTree> mst) {
		this.mst = mst;
	}

	public MinimumSpanningTree get_mst() {
		return _mst;
	}

	public void set_mst(MinimumSpanningTree _mst) {
		this._mst = _mst;
	}

	@Override
	public String toString() {
		return "CoreDistances [idPoint=" + idPoint + ", tmpIdPoint="
				+ tmpIdPoint + ", coreDistance=" + coreDistance + ", point="
				+ Arrays.toString(point) + ", _mst=" + _mst + ", list=" + list
				+ ", graph=" + graph + ", mst=" + mst + "]";
	}
}
