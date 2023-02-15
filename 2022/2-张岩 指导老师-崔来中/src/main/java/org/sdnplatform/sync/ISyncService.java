package org.sdnplatform.sync;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.error.UnknownStoreException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.core.type.TypeReference;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface ISyncService extends IFloodlightService {
    public enum Scope {
        GLOBAL,
        LOCAL,
        UNSYNCHRONIZED
    }
    public short getLocalNodeId();
    public void registerStore(String storeName, Scope scope) 
            throws SyncException;
    public void registerPersistentStore(String storeName, Scope scope) 
            throws SyncException;
    public <K, V> IStoreClient<K, V> getStoreClient(String storeName,
                                                    Class<K> keyClass,
                                                    Class<V> valueClass)
                               throws UnknownStoreException;
    public <K, V> IStoreClient<K, V>
        getStoreClient(String storeName,
                       Class<K> keyClass,
                       Class<V> valueClass,
                       IInconsistencyResolver<Versioned<V>> resolver)
                               throws UnknownStoreException;
    public <K, V> IStoreClient<K, V> getStoreClient(String storeName,
                                                    TypeReference<K> keyType,
                                                    TypeReference<V> valueType)
                               throws UnknownStoreException;
    public <K, V> IStoreClient<K, V>
        getStoreClient(String storeName,
                       TypeReference<K> keyType,
                       TypeReference<V> valueType,
                       IInconsistencyResolver<Versioned<V>> resolver)
                               throws UnknownStoreException;
}
