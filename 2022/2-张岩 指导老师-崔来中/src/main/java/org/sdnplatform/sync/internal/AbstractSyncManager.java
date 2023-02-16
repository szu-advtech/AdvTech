package org.sdnplatform.sync.internal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.sdnplatform.sync.IInconsistencyResolver;
import org.sdnplatform.sync.IStoreClient;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.error.UnknownStoreException;
import org.sdnplatform.sync.internal.store.IStore;
import org.sdnplatform.sync.internal.store.JacksonStore;
import org.sdnplatform.sync.internal.store.MappingStoreListener;
import org.sdnplatform.sync.internal.util.ByteArray;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import com.fasterxml.jackson.core.type.TypeReference;
public abstract class AbstractSyncManager 
    implements ISyncService, IFloodlightModule {
    @Override
    public <K, V> IStoreClient<K, V> 
        getStoreClient(String storeName, 
                       Class<K> keyClass, 
                       Class<V> valueClass)
                               throws  UnknownStoreException {
        return getStoreClient(storeName, keyClass, null, 
                              valueClass, null, null);
    }
    @Override
    public <K, V>IStoreClient<K, V>
        getStoreClient(String storeName, 
                       TypeReference<K> keyType, 
                       TypeReference<V> valueType)
                               throws UnknownStoreException {
        return getStoreClient(storeName, null, keyType, 
                              null, valueType, null);
    }
    @Override
    public <K, V> IStoreClient<K, V>
        getStoreClient(String storeName, 
                       TypeReference<K> keyType, 
                       TypeReference<V> valueType, 
                       IInconsistencyResolver<Versioned<V>> resolver)
                               throws UnknownStoreException {
        return getStoreClient(storeName, null, keyType, 
                              null, valueType, resolver);
    }
    @Override
    public <K, V> IStoreClient<K, V>
        getStoreClient(String storeName, 
                       Class<K> keyClass, 
                       Class<V> valueClass, 
                       IInconsistencyResolver<Versioned<V>> resolver)
                               throws UnknownStoreException {
        return getStoreClient(storeName, keyClass, null,
                              valueClass, null, resolver);
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleServices() {
        Collection<Class<? extends IFloodlightService>> l =
                new ArrayList<Class<? extends IFloodlightService>>();
        l.add(ISyncService.class);
        return l;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        Map<Class<? extends IFloodlightService>,
        IFloodlightService> m =
        new HashMap<Class<? extends IFloodlightService>,
                    IFloodlightService>();
        m.put(ISyncService.class, this);
        return m;
    }
    public <K, V> IStoreClient<K, V>
            getStoreClient(String storeName, 
                           Class<K> keyClass, 
                           TypeReference<K> keyType,
                           Class<V> valueClass, 
                           TypeReference<V> valueType, 
                           IInconsistencyResolver<Versioned<V>> resolver)
                                   throws UnknownStoreException {
        IStore<ByteArray,byte[]> store = getStore(storeName);
        IStore<K, V> serializingStore;
        if (valueType != null && keyType != null) {
            serializingStore = 
                    new JacksonStore<K, V>(store, keyType, valueType);
        } else if (valueClass != null && keyClass != null) {
            serializingStore = 
                    new JacksonStore<K, V>(store, keyClass, valueClass);
        } else {
            throw new IllegalArgumentException("Must include type reference" +
                    " or value class");
        }
        DefaultStoreClient<K, V> storeClient =
                new DefaultStoreClient<K, V>(serializingStore,
                        resolver,
                        this,
                        keyClass,
                        keyType);
        return storeClient;
    }
    public abstract IStore<ByteArray,byte[]> getStore(String storeName)
            throws UnknownStoreException; 
    public abstract short getLocalNodeId();
    public abstract void addListener(String storeName, 
                                     MappingStoreListener listener) 
            throws UnknownStoreException;
    public abstract void shutdown();
}
