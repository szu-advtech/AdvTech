package org.sdnplatform.sync;
import java.util.Iterator;
import java.util.Map.Entry;
import org.sdnplatform.sync.error.ObsoleteVersionException;
import org.sdnplatform.sync.error.SyncException;
public interface IStoreClient<K, V> {
    public V getValue(K key) throws SyncException;
    public V getValue(K key, V defaultValue) throws SyncException;
    public Versioned<V> get(K key) throws SyncException;
    public Versioned<V> get(K key, Versioned<V> defaultValue)
            throws SyncException;
    public IClosableIterator<Entry<K, Versioned<V>>> entries()
            throws SyncException;
    public IVersion put(K key, V value) throws SyncException;
    public IVersion put(K key, Versioned<V> versioned)
            throws SyncException;
    public boolean putIfNotObsolete(K key, Versioned<V> versioned)
            throws SyncException;
    public void delete(K key) throws SyncException;
    public void delete(K key, IVersion version) throws SyncException;
    public void addStoreListener(IStoreListener<K> listener);
}
