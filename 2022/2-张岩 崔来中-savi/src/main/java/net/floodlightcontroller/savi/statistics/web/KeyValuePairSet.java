package net.floodlightcontroller.savi.statistics.web;
import java.util.HashSet;
import java.util.Set;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
@JsonSerialize(using=KeyValuePairSetSerializer.class)
public class KeyValuePairSet {
	Set<KeyValuePair> pairs;
	public KeyValuePairSet() {
		pairs = new HashSet<>();
	}
	public void add(KeyValuePair pair) {
		pairs.add(pair);
	}
	public Set<KeyValuePair> getPairs() {
		return pairs;
	}
}
