package org.sdnplatform.sync.internal.store;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.ISyncService.Scope;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.internal.SyncManager;
import org.sdnplatform.sync.internal.util.ByteArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class SynchronizingStorageEngine extends ListenerStorageEngine {
    protected static Logger logger =
                LoggerFactory.getLogger(SynchronizingStorageEngine.class);
    protected SyncManager syncManager;
    protected Scope scope;
    public SynchronizingStorageEngine(IStorageEngine<ByteArray,
                                                    byte[]> localStorage,
                                      SyncManager syncManager,
                                      IDebugCounterService debugCounter, 
                                      Scope scope) {
        super(localStorage, debugCounter);
        this.localStorage = localStorage;
        this.syncManager = syncManager;
        this.scope = scope;
    }
    @Override
    public void put(ByteArray key, Versioned<byte[]> value)
            throws SyncException {
        super.put(key, value);
        if (!Scope.UNSYNCHRONIZED.equals(scope))
            syncManager.queueSyncTask(this, key, value);
    }
    public Scope getScope() {
        return scope;
    }
}
