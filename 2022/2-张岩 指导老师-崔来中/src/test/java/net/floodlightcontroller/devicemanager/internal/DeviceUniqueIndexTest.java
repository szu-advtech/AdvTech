package net.floodlightcontroller.devicemanager.internal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Iterator;
import org.junit.Test;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import junit.framework.TestCase;
public class DeviceUniqueIndexTest extends TestCase {
    protected Entity e1a;
    protected Entity e1b;
    protected Device d1;
    protected Entity e2;
    protected Entity e2alt;
    protected Entity e3;
    protected Entity e4;
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        e1a = new Entity(MacAddress.of(1L), VlanVid.ofVlan(1), IPv4Address.of(1), IPv6Address.of(1, 1), DatapathId.of(1L), OFPort.of(1), new Date());
        e1b = new Entity(MacAddress.of(1L), VlanVid.ofVlan(2), IPv4Address.of(1), IPv6Address.of(1, 1), DatapathId.of(1L), OFPort.of(1), new Date());
        List<Entity> d1Entities = new ArrayList<Entity>(2);
        d1Entities.add(e1a);
        d1Entities.add(e1b);
        d1 = new Device(null, Long.valueOf(1), null, null, null,
                        d1Entities, null);
        e2 = new Entity(MacAddress.of(2L), VlanVid.ofVlan(2), IPv4Address.of(2), IPv6Address.of(2, 2), DatapathId.of(2L), OFPort.of(2), new Date());
        e2alt = new Entity(MacAddress.of(2L), VlanVid.ofVlan(2), IPv4Address.NONE, IPv6Address.NONE, DatapathId.NONE, OFPort.ZERO, Entity.NO_DATE);
        e3 = new Entity(MacAddress.of(3L), VlanVid.ofVlan(3), IPv4Address.NONE, IPv6Address.NONE, DatapathId.of(3L), OFPort.of(3), new Date());
        e4 = new Entity(MacAddress.of(4L), VlanVid.ofVlan(4), IPv4Address.NONE, IPv6Address.NONE, DatapathId.NONE, OFPort.ZERO, new Date());
    }
    protected void verifyIterator(Set<Long> expected, Iterator<Long> it) {
        HashSet<Long> actual = new HashSet<Long>();
        while (it.hasNext()) {
            actual.add(it.next());
        }
        assertEquals(expected, actual);
    }
    @Test
    public void testDeviceUniqueIndex() {
        DeviceUniqueIndex idx1 = new DeviceUniqueIndex(
                                             EnumSet.of(DeviceField.MAC, 
                                                        DeviceField.VLAN));
        idx1.updateIndex(d1, d1.getDeviceKey());
        idx1.updateIndex(e2, 2L);
        assertEquals(Long.valueOf(1L), idx1.findByEntity(e1a));
        assertEquals(Long.valueOf(1L), idx1.findByEntity(e1b));
        assertEquals(Long.valueOf(2L), idx1.findByEntity(e2));
        assertEquals(Long.valueOf(2L), idx1.findByEntity(e2alt));
        assertEquals(null, idx1.findByEntity(e3));
        assertEquals(null, idx1.findByEntity(e4));
        HashSet<Long> expectedKeys = new HashSet<Long>();
        expectedKeys.add(1L);
        expectedKeys.add(2L);
        verifyIterator(expectedKeys, idx1.getAll());
        verifyIterator(Collections.<Long>singleton(1L), 
                       idx1.queryByEntity(e1a));
        verifyIterator(Collections.<Long>singleton(1L), 
                       idx1.queryByEntity(e1b));
        verifyIterator(Collections.<Long>singleton(2L), 
                       idx1.queryByEntity(e2));
        verifyIterator(Collections.<Long>singleton(2L),
                       idx1.queryByEntity(e2alt));
        assertEquals(false, idx1.queryByEntity(e3).hasNext());
        assertEquals(false, idx1.queryByEntity(e3).hasNext());
        assertEquals(Long.valueOf(1L), idx1.findByEntity(e1a));
        idx1.removeEntity(e1a, 1L); 
        assertEquals(null, idx1.findByEntity(e1a));
        assertEquals(Long.valueOf(1L), idx1.findByEntity(e1b));
        assertEquals(Long.valueOf(2L), idx1.findByEntity(e2));
        idx1.removeEntity(e2);  
        assertEquals(null, idx1.findByEntity(e2));
        assertEquals(Long.valueOf(1L), idx1.findByEntity(e1b));
        DeviceUniqueIndex idx2 = new DeviceUniqueIndex(
                                             EnumSet.of(DeviceField.IPv4,
                                            		 	DeviceField.IPv6,
                                                        DeviceField.SWITCH));
        idx2.updateIndex(e3, 3L);
        assertEquals(Long.valueOf(3L), idx2.findByEntity(e3));
        e3.ipv4Address = IPv4Address.of(3);
        assertEquals(null, idx2.findByEntity(e3));
        idx2.updateIndex(e4, 4L);
        assertEquals(null, idx2.findByEntity(e4));
        Device d4 = new Device(null, 4L, null, null, null,
                               Collections.<Entity>singleton(e4), null);
        idx2.updateIndex(d4, 4L);
        assertEquals(null, idx2.findByEntity(e4));
        DeviceUniqueIndex idx3 = new DeviceUniqueIndex(
                                             EnumSet.of(DeviceField.MAC, 
                                                        DeviceField.VLAN));
        idx3.updateIndex(e1a, 42L);
        assertEquals(false, idx3.updateIndex(d1, 1L));
        idx3.updateIndex(e1a, 1L);
        assertEquals(true, idx3.updateIndex(d1, 1L));
    }
}