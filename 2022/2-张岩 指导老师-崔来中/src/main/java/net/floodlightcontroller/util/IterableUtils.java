package net.floodlightcontroller.util;
import java.util.ArrayList;
import java.util.Collection;
public class IterableUtils {
	public static <T> Collection<T> toCollection(Iterable<T> i) {
		if (i == null) {
			throw new IllegalArgumentException("Iterable 'i' cannot be null");
		}
		Collection<T> c = new ArrayList<T>();
		for (T t : i) {
			c.add(t);
		}
		return c;
	}
}