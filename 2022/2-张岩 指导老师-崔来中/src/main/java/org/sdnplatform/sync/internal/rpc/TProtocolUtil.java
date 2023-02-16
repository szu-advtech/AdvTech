package org.sdnplatform.sync.internal.rpc;
import java.util.ArrayList;
import java.util.List;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.ISyncService.Scope;
import org.sdnplatform.sync.internal.util.ByteArray;
import org.sdnplatform.sync.internal.version.ClockEntry;
import org.sdnplatform.sync.internal.version.VectorClock;
import org.sdnplatform.sync.thrift.AsyncMessageHeader;
import org.sdnplatform.sync.thrift.SyncMessage;
import org.sdnplatform.sync.thrift.KeyedValues;
import org.sdnplatform.sync.thrift.KeyedVersions;
import org.sdnplatform.sync.thrift.MessageType;
import org.sdnplatform.sync.thrift.Store;
import org.sdnplatform.sync.thrift.SyncOfferMessage;
import org.sdnplatform.sync.thrift.SyncValueMessage;
import org.sdnplatform.sync.thrift.VersionedValue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class TProtocolUtil {
    protected static Logger logger =
            LoggerFactory.getLogger(TProtocolUtil.class.getName());
    public static org.sdnplatform.sync.thrift.VectorClock
        getTVectorClock(VectorClock vc) {
        org.sdnplatform.sync.thrift.VectorClock tvc =
                new org.sdnplatform.sync.thrift.VectorClock();
        tvc.setTimestamp(vc.getTimestamp());
        for (ClockEntry ce : vc.getEntries()) {
            org.sdnplatform.sync.thrift.ClockEntry tce =
                    new org.sdnplatform.sync.thrift.ClockEntry();
            tce.setNodeId(ce.getNodeId());
            tce.setVersion(ce.getVersion());
            tvc.addToVersions(tce);
        }
        return tvc;
    }
    public static org.sdnplatform.sync.thrift.VersionedValue
        getTVersionedValue(Versioned<byte[]> value) {
        org.sdnplatform.sync.thrift.VersionedValue tvv =
                new org.sdnplatform.sync.thrift.VersionedValue();
        org.sdnplatform.sync.thrift.VectorClock tvc = 
                getTVectorClock((VectorClock)value.getVersion());
        tvv.setVersion(tvc);
        tvv.setValue(value.getValue());
        return tvv;
    }
    @SafeVarargs
	public static KeyedValues getTKeyedValues(ByteArray key, 
                                              Versioned<byte[]>... value) {
        KeyedValues kv = new KeyedValues();
        kv.setKey(key.get());
        for (Versioned<byte[]> v : value) {
            kv.addToValues(getTVersionedValue(v));
        }
        return kv;
    }
    public static KeyedValues 
            getTKeyedValues(ByteArray key, 
                            Iterable<Versioned<byte[]>> values) {
        KeyedValues kv = new KeyedValues();
        kv.setKey(key.get());
        for (Versioned<byte[]> v : values) {
            kv.addToValues(getTVersionedValue(v));
        }
        return kv;
    }
    public static KeyedVersions 
            getTKeyedVersions(ByteArray key, List<Versioned<byte[]>> values) {
        KeyedVersions kv = new KeyedVersions();
        kv.setKey(key.get());
        for (Versioned<byte[]> v : values) {
            kv.addToVersions(getTVectorClock((VectorClock)v.getVersion()));
        }
        return kv;
    }
    public static org.sdnplatform.sync.thrift.Store getTStore(String storeName,
                                                               Scope scope, 
                                                               boolean persist) {
        return getTStore(storeName, getTScope(scope), persist);
    }
    public static org.sdnplatform.sync.thrift.Store 
            getTStore(String storeName,
                      org.sdnplatform.sync.thrift.Scope scope,
                      boolean persist) {
        org.sdnplatform.sync.thrift.Store store =
                new org.sdnplatform.sync.thrift.Store();
        store.setScope(scope);
        store.setStoreName(storeName);
        store.setPersist(persist);
        return store;
    }
    public static Scope getScope(org.sdnplatform.sync.thrift.Scope tScope) {
        switch (tScope) {
            case LOCAL:
                return Scope.LOCAL;
            case UNSYNCHRONIZED:
                return Scope.UNSYNCHRONIZED;                
            case GLOBAL:
            default:
                return Scope.GLOBAL;
        }
    }
    public static org.sdnplatform.sync.thrift.Scope getTScope(Scope Scope) {
        switch (Scope) {
            case LOCAL:
                return org.sdnplatform.sync.thrift.Scope.LOCAL;
            case UNSYNCHRONIZED:
                return org.sdnplatform.sync.thrift.Scope.UNSYNCHRONIZED;
            case GLOBAL:
            default:
                return org.sdnplatform.sync.thrift.Scope.GLOBAL;
        }
    }
    public static SyncMessage getTSyncValueMessage(String storeName, 
                                                      Scope scope,
                                                      boolean persist) {
        return getTSyncValueMessage(getTStore(storeName, scope, persist));
    }
    public static SyncMessage getTSyncValueMessage(Store store) {
        SyncMessage bsm = 
                new SyncMessage(MessageType.SYNC_VALUE);
        AsyncMessageHeader header = new AsyncMessageHeader();
        SyncValueMessage svm = new SyncValueMessage();
        svm.setHeader(header);
        svm.setStore(store);
        bsm.setSyncValue(svm);
        return bsm;
    }
    public static SyncMessage getTSyncOfferMessage(String storeName,
                                                      Scope scope,
                                                      boolean persist) {
        SyncMessage bsm = new SyncMessage(MessageType.SYNC_OFFER);
        AsyncMessageHeader header = new AsyncMessageHeader();
        SyncOfferMessage som = new SyncOfferMessage();
        som.setHeader(header);
        som.setStore(getTStore(storeName, scope, persist));
        bsm.setSyncOffer(som);
        return bsm;
    }
    public static VectorClock getVersion(org.sdnplatform.sync.thrift.VectorClock tvc) {
        ArrayList<ClockEntry> entries =
                new ArrayList<ClockEntry>();
        if (tvc.getVersions() != null) {
            for (org.sdnplatform.sync.thrift.ClockEntry ce :
                tvc.getVersions()) {
                entries.add(new ClockEntry(ce.getNodeId(), ce.getVersion()));
            }
        }
        return new VectorClock(entries, tvc.getTimestamp());
    }
    public static Versioned<byte[]> 
            getVersionedValued(VersionedValue tvv) {
                Versioned<byte[]> vv =
                new Versioned<byte[]>(tvv.getValue(), 
                                      getVersion(tvv.getVersion()));
        return vv;
    }
    public static List<Versioned<byte[]>> getVersionedList(List<VersionedValue> tvv) {
        ArrayList<Versioned<byte[]>> values = 
                new ArrayList<Versioned<byte[]>>();
        if (tvv != null) {
            for (VersionedValue v : tvv) {
                values.add(TProtocolUtil.getVersionedValued(v));
            }
        }
        return values;
    }
}
