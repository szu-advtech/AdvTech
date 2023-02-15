package net.floodlightcontroller.savi.statistics.web;
public class KeyValuePair {
	protected String key;
	protected String value;
	public KeyValuePair(String key, String value) {
		this.key = key;
		this.value = value;
	}
	public String getKey() {
		return key;
	}
	public void setKey(String key) {
		this.key = key;
	}
	public String getValue() {
		return value;
	}
	public void setValue(String value) {
		this.value = value;
	}
}
