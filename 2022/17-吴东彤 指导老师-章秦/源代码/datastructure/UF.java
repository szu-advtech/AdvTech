package datastructure;

import java.io.Serializable;

public class UF implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int[] id;

	public UF(int n) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		id = new int[n];
		for (int i = 0; i < n; i++) {
			id[i] = i;
		}
	}

	public int[] getId() {
		return id;
	}

	public void setId(int[] id) {
		this.id = id;
	}

	public int[] union(int p, int q) {
		int pid = p;
		int qid = q;

		for (int i = 0; i < id.length; i++) {
			if (id[i] == pid) {
				id[i] = qid;
			}
		}
		return id;
	}

	public int find(int p) {
		return id[p];
	}

	public boolean connected(int p, int q) {
		return find(p) == find(q);
	}
}
