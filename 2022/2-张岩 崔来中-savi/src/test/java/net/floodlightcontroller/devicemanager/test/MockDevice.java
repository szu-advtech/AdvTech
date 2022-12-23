package net.floodlightcontroller.devicemanager.test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.TreeSet;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.devicemanager.internal.AttachmentPoint;
import net.floodlightcontroller.devicemanager.internal.Device;
import net.floodlightcontroller.devicemanager.internal.DeviceManagerImpl;
import net.floodlightcontroller.devicemanager.internal.Entity;
public class MockDevice extends Device {
    public MockDevice(DeviceManagerImpl deviceManager,
                      Long deviceKey,
                      Entity entity, 
                      IEntityClass entityClass)  {
        super(deviceManager, deviceKey, entity, entityClass);
    }
    public MockDevice(Device device, Entity newEntity, int insertionpoint) {
        super(device, newEntity, insertionpoint);
    }
    public MockDevice(DeviceManagerImpl deviceManager, Long deviceKey,
                      List<AttachmentPoint> aps,
                      List<AttachmentPoint> trueAPs,
                      Collection<Entity> entities,
                      IEntityClass entityClass) {
        super(deviceManager, deviceKey, null, aps, trueAPs,
              entities, entityClass);
    }
    @Override
    public IPv4Address[] getIPv4Addresses() {
        TreeSet<IPv4Address> vals = new TreeSet<IPv4Address>();
        for (Entity e : entities) {
            if (e.getIpv4Address().equals(IPv4Address.NONE)) continue;
            vals.add(e.getIpv4Address());
        }
        return vals.toArray(new IPv4Address[vals.size()]);
    }
    @Override
    public IPv6Address[] getIPv6Addresses() {
        TreeSet<IPv6Address> vals = new TreeSet<IPv6Address>();
        for (Entity e : entities) {
            if (e.getIpv6Address().equals(IPv6Address.NONE)) continue;
            vals.add(e.getIpv6Address());
        }
        return vals.toArray(new IPv6Address[vals.size()]);
    }
    @Override
    public SwitchPort[] getAttachmentPoints() {
        ArrayList<SwitchPort> vals = 
                new ArrayList<SwitchPort>(entities.length);
        for (Entity e : entities) {
            if (!e.getSwitchDPID().equals(DatapathId.NONE) &&
                !e.getSwitchPort().equals(OFPort.ZERO) &&
                deviceManager.isValidAttachmentPoint(e.getSwitchDPID(), e.getSwitchPort())) {
                SwitchPort sp = new SwitchPort(e.getSwitchDPID(), 
                                               e.getSwitchPort());
                vals.add(sp);
            }
        }
        return vals.toArray(new SwitchPort[vals.size()]);
    }
    @Override
    public String toString() {
        return "MockDevice [getEntityClass()=" + getEntityClass()
               + ", getEntities()=" + Arrays.toString(getEntities()) + "]";
    }
}