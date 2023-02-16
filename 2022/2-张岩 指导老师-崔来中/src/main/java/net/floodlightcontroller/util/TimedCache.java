package net.floodlightcontroller.util;
import com.googlecode.concurrentlinkedhashmap.ConcurrentLinkedHashMap;
import java.util.concurrent.ConcurrentMap;
public class TimedCache<K> {    
	private ConcurrentMap<K, Long> cache;
	public TimedCache(int capacity, int timeToLive) {
        cache = new ConcurrentLinkedHashMap.Builder<K, Long>()
        	    .maximumWeightedCapacity(capacity)
            .build();
        this.timeoutInterval = timeToLive;
    }
    public long getTimeoutInterval() {
        return this.timeoutInterval;
    }
    public boolean update(K key)
    {
        Long curr = new Long(System.currentTimeMillis());
        Long prev = cache.putIfAbsent(key, curr);
        if (prev == null) {
        		return false;
        }
        if (curr - prev > this.timeoutInterval) {
            if (cache.replace(key, prev, curr)) {
            		return false;
            }
        }
        return true;
    }
}
