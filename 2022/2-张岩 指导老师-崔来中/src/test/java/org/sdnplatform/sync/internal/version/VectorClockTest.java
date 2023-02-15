package org.sdnplatform.sync.internal.version;
import static org.sdnplatform.sync.internal.TUtils.getClockT;
import org.junit.Test;
import org.sdnplatform.sync.IVersion.Occurred;
import org.sdnplatform.sync.internal.version.ClockEntry;
import org.sdnplatform.sync.internal.version.VectorClock;
import com.google.common.collect.Lists;
public class VectorClockTest {
    @Test
    public void testEqualsAndHashcode() {
        long now = 5555555555L;
        VectorClock one = getClockT(now, 1, 2);
        VectorClock other = getClockT(now, 1, 2);
        assertEquals(one, other);
        assertEquals(one.hashCode(), other.hashCode());
    }
    @Test
    public void testComparisons() {
        assertTrue("The empty clock should not happen before itself.",
                   getClock().compare(getClock()) != Occurred.CONCURRENTLY);
        assertTrue("A clock should not happen before an identical clock.",
                   getClock(1, 1, 2).compare(getClock(1, 1, 2)) != Occurred.CONCURRENTLY);
        assertTrue(" A clock should happen before an identical clock with a single additional event.",
                   getClock(1, 1, 2).compare(getClock(1, 1, 2, 3)) == Occurred.BEFORE);
        assertTrue("Clocks with different events should be concurrent.",
                   getClock(1).compare(getClock(2)) == Occurred.CONCURRENTLY);
        assertTrue("Clocks with different events should be concurrent.",
                   getClock(1, 1, 2).compare(getClock(1, 1, 3)) == Occurred.CONCURRENTLY);
        assertTrue(getClock(2, 2).compare(getClock(1, 2, 2, 3)) == Occurred.BEFORE
                   && getClock(1, 2, 2, 3).compare(getClock(2, 2)) == Occurred.AFTER);
    }
    @Test
    public void testMerge() {
        assertEquals("Two empty clocks merge to an empty clock.",
                     getClock().merge(getClock()).getEntries(),
                     getClock().getEntries());
        assertEquals("Merge of a clock with itself does nothing",
                     getClock(1).merge(getClock(1)).getEntries(),
                     getClock(1).getEntries());
        assertEquals(getClock(1).merge(getClock(2)).getEntries(), getClock(1, 2).getEntries());
        assertEquals(getClock(1).merge(getClock(1, 2)).getEntries(), getClock(1, 2).getEntries());
        assertEquals(getClock(1, 2).merge(getClock(1)).getEntries(), getClock(1, 2).getEntries());
        assertEquals("Two-way merge fails.",
                     getClock(1, 1, 1, 2, 3, 5).merge(getClock(1, 2, 2, 4)).getEntries(),
                     getClock(1, 1, 1, 2, 2, 3, 4, 5).getEntries());
        assertEquals(getClock(2, 3, 5).merge(getClock(1, 2, 2, 4, 7)).getEntries(),
                     getClock(1, 2, 2, 3, 4, 5, 7).getEntries());
    }
    @Test
    public void testMergeWithLargeVersion() {
        VectorClock clock1 = getClock(1);
        VectorClock clock2 = new VectorClock(Lists.newArrayList(new ClockEntry((short) 1,
                                                                               Short.MAX_VALUE + 1)),
                                             System.currentTimeMillis());
        VectorClock mergedClock = clock1.merge(clock2);
        assertEquals(mergedClock.getMaxVersion(), Short.MAX_VALUE + 1);
    }
    @Test
    public void testIncrementOrderDoesntMatter() {
        int numTests = 10;
        int numNodes = 10;
        int numValues = 100;
        VectorClock[] clocks = new VectorClock[numNodes];
        for(int t = 0; t < numTests; t++) {
            int[] test = randomInts(numNodes, numValues);
            for(int n = 0; n < numNodes; n++)
                clocks[n] = getClock(shuffle(test));
            for(int n = 0; n < numNodes - 1; n++)
                assertEquals("Clock " + n + " and " + (n + 1) + " are not equal.",
                             clocks[n].getEntries(),
                             clocks[n + 1].getEntries());
        }
    }
    public void testIncrementAndSerialize() {
        int node = 1;
        VectorClock vc = getClock(node);
        assertEquals(node, vc.getMaxVersion());
        int increments = 3000;
        for(int i = 0; i < increments; i++) {
            vc.incrementVersion(node, 45);
            vc = new VectorClock(vc.toBytes());
        }
        assertEquals(increments + 1, vc.getMaxVersion());
}
