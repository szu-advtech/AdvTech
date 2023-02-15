package org.sdnplatform.sync.internal.store;
import java.util.List;
import java.util.Map.Entry;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.error.SyncException;
public interface IStorageEngine<K, V> extends IStore<K, V> {
    public IClosableIterator<Entry<K,List<Versioned<V>>>> entries();
    public IClosableIterator<K> keys();
    public void truncate() throws SyncException;
    public boolean writeSyncValue(K key, Iterable<Versioned<V>> values);
    public void cleanupTask() throws SyncException;
    public boolean isPersistent();
    void setTombstoneInterval(int interval);
}
