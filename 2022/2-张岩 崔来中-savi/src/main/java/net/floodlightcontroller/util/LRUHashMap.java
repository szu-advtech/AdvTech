package net.floodlightcontroller.util;
import java.util.LinkedHashMap;
import java.util.Map;
public class LRUHashMap<K, V> extends LinkedHashMap<K, V> {
    private static final long serialVersionUID = 1L;
    private final int capacity;
    public LRUHashMap(int capacity)
    {
        super(capacity+1, 0.75f, true);
        this.capacity = capacity;
    }
    protected boolean removeEldestEntry(Map.Entry<K,V> eldest) {
       return size() > capacity;
    }
}
