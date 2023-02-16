package net.floodlightcontroller.devicemanager.internal;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.util.FilterIterator;
public class DeviceIterator extends FilterIterator<Device> {
    private IEntityClass[] entityClasses;
    private MacAddress macAddress;
    private VlanVid vlan;
    private IPv4Address ipv4Address; 
    private IPv6Address ipv6Address;
    private DatapathId switchDPID;
    private OFPort switchPort;
    public DeviceIterator(Iterator<Device> subIterator, 
                          IEntityClass[] entityClasses,
                          MacAddress macAddress,
                          VlanVid vlan, 
                          IPv4Address ipv4Address, 
                          IPv6Address ipv6Address,
                          DatapathId switchDPID,
                          OFPort switchPort) {
        super(subIterator);
        this.entityClasses = entityClasses;
        this.subIterator = subIterator;
        this.macAddress = macAddress;
        this.vlan = vlan;
        this.ipv4Address = ipv4Address;
        this.ipv6Address = ipv6Address;
        this.switchDPID = switchDPID;
        this.switchPort = switchPort;
    }
    @Override
    protected boolean matches(Device value) {
        boolean match;
        if (entityClasses != null) {
            IEntityClass clazz = value.getEntityClass();
            if (clazz == null) return false;
            match = false;
            for (IEntityClass entityClass : entityClasses) {
                if (clazz.equals(entityClass)) {
                    match = true;
                    break;
                }
            }
            if (!match) return false;                
        }
        if (!macAddress.equals(MacAddress.NONE)) {
            if (!macAddress.equals(value.getMACAddress()))
                return false;
        }
            VlanVid[] vlans = value.getVlanId();
            List<VlanVid> searchableVlanList = Arrays.asList(vlans);
            if (!searchableVlanList.contains(vlan)) {
            	return false;
            }
        }
        if (!ipv4Address.equals(IPv4Address.NONE)) {
            IPv4Address[] ipv4Addresses = value.getIPv4Addresses();
            List<IPv4Address> searchableIPv4AddrList = Arrays.asList(ipv4Addresses);
            if (!searchableIPv4AddrList.contains(ipv4Address)) {
            	return false;
            }
        }
        if (!ipv6Address.equals(IPv6Address.NONE)) {
            IPv6Address[] ipv6Addresses = value.getIPv6Addresses();
            List<IPv6Address> searchableIPv6AddrList = Arrays.asList(ipv6Addresses);
            if (!searchableIPv6AddrList.contains(ipv6Address)) {
            	return false;
            }
        }
        if (!switchDPID.equals(DatapathId.NONE) || !switchPort.equals(OFPort.ZERO)) {
            SwitchPort[] sps = value.getAttachmentPoints();
            if (sps == null) return false;
            match = false;
            for (SwitchPort sp : sps) {
                if (!switchDPID.equals(DatapathId.NONE)) {
                    if (!switchDPID.equals(sp.getSwitchDPID()))
                        return false;
                }
                if (!switchPort.equals(OFPort.ZERO)) {
                    if (!switchPort.equals(sp.getPort()))
                        return false;
                }
                match = true;
                break;
            }
            if (!match) return false;
        }
        return true;
    }
}