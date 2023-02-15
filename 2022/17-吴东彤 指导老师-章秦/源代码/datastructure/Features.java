package datastructure;

import java.util.Arrays;

public class Features {
      private double[][] ls;
      private double[][] ss;
      private int[] n;
      private int[][] indices;
      
      public Features(){
    	  
      }
      
      public Features(int[][] indices, double[][] ls, double[][] ss, int[] n){
    	  this.setIndices(indices);
    	  this.setLs(ls);
    	  this.setSs(ss);
    	  this.setN(n);
      }

	public double[][] getLs() {
		return ls;
	}

	public void setLs(double[][] ls) {
		this.ls = ls;
	}

	public double[][] getSs() {
		return ss;
	}

	public void setSs(double[][] ss) {
		this.ss = ss;
	}

	public int[] getN() {
		return n;
	}

	public void setN(int[] n) {
		this.n = n;
	}

	@Override
	public String toString() {
		return "Features [ls=" + Arrays.toString(ls) + ", ss="
				+ Arrays.toString(ss) + ", n=" + Arrays.toString(n) + "]";
	}

	public int[][] getIndices() {
		return indices;
	}

	public void setIndices(int[][] indices) {
		this.indices = indices;
	}  
}
