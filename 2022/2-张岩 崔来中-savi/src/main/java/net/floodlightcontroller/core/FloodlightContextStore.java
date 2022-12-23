package net.floodlightcontroller.core;
public class FloodlightContextStore<V> {
    @SuppressWarnings("unchecked")
    public V get(FloodlightContext bc, String key) {
        return (V)bc.storage.get(key);
    }
    public void put(FloodlightContext bc, String key, V value) {
        bc.storage.put(key, value);
    }
    public void remove(FloodlightContext bc, String key) {
        bc.storage.remove(key);
    }
}
