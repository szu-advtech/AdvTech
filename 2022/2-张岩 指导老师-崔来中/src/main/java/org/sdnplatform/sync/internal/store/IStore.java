package org.sdnplatform.sync.internal.store;
import java.util.List;
import java.util.Map.Entry;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.IVersion;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.error.SyncException;
public interface IStore<K, V> {
    public List<Versioned<V>> get(K key) throws SyncException;
    public IClosableIterator<Entry<K,List<Versioned<V>>>> entries();
    public void put(K key, Versioned<V> value)
            throws SyncException;
    public List<IVersion> getVersions(K key) throws SyncException;
    public String getName();
    public void close() throws SyncException;
}
