package datastructure;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

public class ListMST extends LinkedList<String> implements Serializable{

	private static final long serialVersionUID = 1L;
	
	@Override
	public String toString(){
		Iterator<String> it = iterator();
	    if (! it.hasNext())
	        return "[]";

	    StringBuilder sb = new StringBuilder();
	    for (;;) {
	        String e = it.next();
	        sb.append(e);
	        if (! it.hasNext())
	            return sb.append(' ').toString();
	    }
	}
}
