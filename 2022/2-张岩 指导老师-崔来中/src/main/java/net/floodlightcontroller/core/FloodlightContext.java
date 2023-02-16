package net.floodlightcontroller.core;
import java.util.concurrent.ConcurrentHashMap;
public class FloodlightContext {
    protected ConcurrentHashMap<String, Object> storage =
            new ConcurrentHashMap<String, Object>();
    public ConcurrentHashMap<String, Object> getStorage() {
        return storage;
    }
}
