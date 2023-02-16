package org.sdnplatform.sync.internal.version;
import org.junit.Test;
import org.sdnplatform.sync.internal.version.ClockEntry;
public class ClockEntryTest {
    @Test
    public void testEquality() {
        ClockEntry v1 = new ClockEntry((short) 0, 1);
        ClockEntry v2 = new ClockEntry((short) 0, 1);
        assertTrue(v1.equals(v1));
        assertTrue(!v1.equals(null));
        assertEquals(v1, v2);
        v1 = new ClockEntry((short) 0, 1);
        v2 = new ClockEntry((short) 0, 2);
        assertTrue(!v1.equals(v2));
        v1 = new ClockEntry(Short.MAX_VALUE, 256);
        v2 = new ClockEntry(Short.MAX_VALUE, 256);
        assertEquals(v1, v2);
    }
    @Test
    public void testIncrement() {
        ClockEntry v = new ClockEntry((short) 0, 1);
        assertEquals(v.getNodeId(), 0);
        assertEquals(v.getVersion(), 1);
        ClockEntry v2 = v.incremented();
        assertEquals(v.getVersion(), 1);
        assertEquals(v2.getVersion(), 2);
    }
}
