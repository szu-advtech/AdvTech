package org.sdnplatform.sync.internal.store;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.IVersion;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.IVersion.Occurred;
import org.sdnplatform.sync.error.ObsoleteVersionException;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.internal.util.Pair;
public class InMemoryStorageEngine<K, V> implements IStorageEngine<K, V> {
    private final ConcurrentMap<K, List<Versioned<V>>> map;
    private final String name;
    public InMemoryStorageEngine(String name) {
        this.name = name;
        this.map = new ConcurrentHashMap<K, List<Versioned<V>>>();
    }
    public InMemoryStorageEngine(String name, 
                                 ConcurrentMap<K, List<Versioned<V>>> map) {
        this.name = name;
        this.map = map;
    }
    @Override
    public void close() {}
    @Override
    public List<IVersion> getVersions(K key) throws SyncException {
        return StoreUtils.getVersions(get(key));
    }
    @Override
    public List<Versioned<V>> get(K key) throws SyncException {
        StoreUtils.assertValidKey(key);
        List<Versioned<V>> results = map.get(key);
        if(results == null) {
            return new ArrayList<Versioned<V>>(0);
        }
        synchronized(results) {
            return new ArrayList<Versioned<V>>(results);
        }
    }
    @Override
    public void put(K key, Versioned<V> value) throws SyncException {
        if (!doput(key, value))
            throw new ObsoleteVersionException();
    }
    public boolean doput(K key, Versioned<V> value) throws SyncException {
        StoreUtils.assertValidKey(key);
        IVersion version = value.getVersion();
        while(true) {
            List<Versioned<V>> items = map.get(key);
            if(items == null) {
                items = new ArrayList<Versioned<V>>();
                items.add(new Versioned<V>(value.getValue(), version));
                if (map.putIfAbsent(key, items) != null)
                    continue;
                return true;
            } else {
                synchronized(items) {
                    if(map.get(key) != items)
                        continue;
                    List<Versioned<V>> itemsToRemove = new ArrayList<Versioned<V>>(items.size());
                    for(Versioned<V> versioned: items) {
                        Occurred occurred = value.getVersion().compare(versioned.getVersion());
                        if(occurred == Occurred.BEFORE) {
                            return false;
                        } else if(occurred == Occurred.AFTER) {
                            itemsToRemove.add(versioned);
                        }
                    }
                    items.removeAll(itemsToRemove);
                    items.add(value);
                }
                return true;
            }
        }
    }
    @Override
    public IClosableIterator<Entry<K,List<Versioned<V>>>> entries() {
        return new InMemoryIterator<K, V>(map);
    }
    @Override
    public IClosableIterator<K> keys() {
        return StoreUtils.keys(entries());
    }
    @Override
    public void truncate() {
        map.clear();
    }
    @Override
    public String getName() {
        return name;
    }
    @Override
    public boolean writeSyncValue(K key, Iterable<Versioned<V>> values) {
        boolean success = false;
        for (Versioned<V> value : values) {
            try {
                put (key, value);
                success = true;
            } catch (SyncException e) {
            }
        }
        return success;
    }
    @Override
    public void cleanupTask() {
        Iterator<Entry<K, List<Versioned<V>>>> iter = map.entrySet().iterator();
        while (iter.hasNext()) {
            Entry<K, List<Versioned<V>>> e = iter.next();
            List<Versioned<V>> items = e.getValue();
            synchronized (items) {
                if (StoreUtils.canDelete(items, tombstoneDeletion))
                    iter.remove();
            }
        }
    }
    @Override
    public boolean isPersistent() {
        return false;
    }
    @Override
    public void setTombstoneInterval(int interval) {
        this.tombstoneDeletion = interval;
    }
    public int size() {
        return map.size();
    }
    public List<Versioned<V>> remove(K key) {
        while (true) {
            List<Versioned<V>> items = map.get(key);
            synchronized (items) {
                if (map.remove(key, items))
                    return items;                
            }
        }
    }
    public boolean containsKey(K key) {
        return map.containsKey(key);
    }
    @Override
    public String toString() {
        return toString(15);
    }
    protected String toString(int size) {
        StringBuilder builder = new StringBuilder();
        builder.append("{");
        int count = 0;
        for(Entry<K, List<Versioned<V>>> entry: map.entrySet()) {
            if(count > size) {
                builder.append("...");
                break;
            }
            builder.append(entry.getKey());
            builder.append(':');
            builder.append(entry.getValue());
            builder.append(',');
        }
        builder.append('}');
        return builder.toString();
    }
    private static class InMemoryIterator<K, V> implements 
        IClosableIterator<Entry<K, List<Versioned<V>>>> {
        private final Iterator<Entry<K, List<Versioned<V>>>> iterator;
        public InMemoryIterator(ConcurrentMap<K, List<Versioned<V>>> map) {
            this.iterator = map.entrySet().iterator();
        }
        public boolean hasNext() {
            return iterator.hasNext();
        }
        public Pair<K, List<Versioned<V>>> next() {
            Entry<K, List<Versioned<V>>> entry = iterator.next();
            return new Pair<K, List<Versioned<V>>>(entry.getKey(), 
                    entry.getValue());
        }
        public void remove() {
            throw new UnsupportedOperationException("No removal y'all.");
        }
        @Override
        public void close() {
        }
    }
}
