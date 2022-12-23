package org.sdnplatform.sync.internal.store;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.sdnplatform.sync.internal.TUtils;
import org.sdnplatform.sync.internal.store.IStorageEngine;
import org.sdnplatform.sync.internal.store.InMemoryStorageEngine;
import org.sdnplatform.sync.internal.util.ByteArray;
public class InMemoryStorageEngineTest extends AbstractStorageEngineT {
    private IStorageEngine<ByteArray, byte[]> store;
    @Override
    public IStorageEngine<ByteArray, byte[]> getStorageEngine() {
        return store;
    }
    @Before
    public void setUp() throws Exception {
        this.store = new InMemoryStorageEngine<ByteArray, byte[]>("test");
    }
    @Override
    public List<ByteArray> getKeys(int numKeys) {
        List<ByteArray> keys = new ArrayList<ByteArray>(numKeys);
        for(int i = 0; i < numKeys; i++)
            keys.add(new ByteArray(TUtils.randomBytes(10)));
        return keys;
    }
}
