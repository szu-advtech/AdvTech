package org.sdnplatform.sync;
import java.util.Iterator;
public interface IStoreListener<K> {
    public enum UpdateType {
        LOCAL,
        REMOTE
    };
    public void keysModified(Iterator<K> keys, UpdateType type);
}
