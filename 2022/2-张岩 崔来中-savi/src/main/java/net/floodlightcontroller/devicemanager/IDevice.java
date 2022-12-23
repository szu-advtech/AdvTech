package net.floodlightcontroller.devicemanager;
import java.util.Date;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.VlanVid;
public interface IDevice {
    public Long getDeviceKey();
    public MacAddress getMACAddress();
    public String getMACAddressString();
    public VlanVid[] getVlanId();
    public IPv4Address[] getIPv4Addresses();
    public IPv6Address[] getIPv6Addresses();
    public SwitchPort[] getAttachmentPoints();
    public SwitchPort[] getOldAP();
    public SwitchPort[] getAttachmentPoints(boolean includeError);
    public VlanVid[] getSwitchPortVlanIds(SwitchPort swp);
    public Date getLastSeen();
    public IEntityClass getEntityClass();
}
