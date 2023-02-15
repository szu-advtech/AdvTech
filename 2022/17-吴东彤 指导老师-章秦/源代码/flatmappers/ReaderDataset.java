package flatmappers;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.spark.api.java.function.*;

public class ReaderDataset implements FlatMapFunction<String, String>, Serializable{
	/**
	 * 
	 */
	public ReaderDataset(){
		
	}
	private static final long serialVersionUID = 1L;

	public Iterator<String> call(String s) {
		return Arrays.asList(s.split("\n")).iterator();
	}
}
