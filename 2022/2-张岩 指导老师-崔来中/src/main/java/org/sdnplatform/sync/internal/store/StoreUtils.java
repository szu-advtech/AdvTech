package org.sdnplatform.sync.internal.store;
import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.IVersion;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.IVersion.Occurred;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.internal.version.VectorClock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
public class StoreUtils {
    protected static final Logger logger =
            LoggerFactory.getLogger(StoreUtils.class);
    public static void assertValidKeys(Iterable<?> keys) {
        if(keys == null)
            throw new IllegalArgumentException("Keys cannot be null.");
        for(Object key: keys)
            assertValidKey(key);
    }
    public static <K> void assertValidKey(K key) {
        if(key == null)
            throw new IllegalArgumentException("Key cannot be null.");
    }
    public static <K, V> Map<K, List<Versioned<V>>>
        getAll(IStore<K, V> storageEngine,
               Iterable<K> keys) throws SyncException {
        Map<K, List<Versioned<V>>> result = newEmptyHashMap(keys);
        for(K key: keys) {
            List<Versioned<V>> value =
                    storageEngine.get(key);
            if(!value.isEmpty())
                result.put(key, value);
        }
        return result;
    }
    public static <K, V> HashMap<K, V> newEmptyHashMap(Iterable<?> iterable) {
        if(iterable instanceof Collection<?>)
            return Maps.newHashMapWithExpectedSize(((Collection<?>) iterable).size());
        return Maps.newHashMap();
    }
    public static void close(Closeable c) {
        if(c != null) {
            try {
                c.close();
            } catch(IOException e) {
                logger.error("Error closing stream", e);
            }
        }
    }
    public static <V> List<IVersion> getVersions(List<Versioned<V>> versioneds) {
        List<IVersion> versions = Lists.newArrayListWithCapacity(versioneds.size());
        for(Versioned<?> versioned: versioneds)
            versions.add(versioned.getVersion());
        return versions;
    }
    public static <K, V> IClosableIterator<K>
        keys(final IClosableIterator<Entry<K, V>> values) {
        return new IClosableIterator<K>() {
            public void close() {
                values.close();
            }
            public boolean hasNext() {
                return values.hasNext();
            }
            public K next() {
                Entry<K, V> value = values.next();
                if(value == null)
                    return null;
                return value.getKey();
            }
            public void remove() {
                values.remove();
            }
        };
    }
    public static <V> boolean canDelete(List<Versioned<V>> items,
                                         long tombstoneDeletion) {
        List<VectorClock> tombstones = new ArrayList<VectorClock>();
        long now = System.currentTimeMillis();
        for (Versioned<V> v : items) {
            if (v.getValue() == null) {
                VectorClock vc = (VectorClock)v.getVersion();
                if ((vc.getTimestamp() + tombstoneDeletion) < now)
                    tombstones.add(vc);
            }
        }
        for (VectorClock vc : tombstones) {
            boolean later = true;
            for (Versioned<V> v : items) {
                if (v.getValue() != null) {
                    VectorClock curvc = (VectorClock)v.getVersion();
                    if (!Occurred.AFTER.equals(vc.compare(curvc))) {
                        later = false;
                        break;
                    }
                }
            }
            if (later) {
                return true;
            }
        }
        return false;
    }
}
