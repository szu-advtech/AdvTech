package net.floodlightcontroller.devicemanager;
import java.util.Collection;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.Set;
import javax.annotation.Nonnull;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.VlanVid;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.FloodlightContextStore;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IDeviceService extends IFloodlightService {
    enum DeviceField {
        MAC, IPv4, IPv6, VLAN, SWITCH, PORT
    }
    public static final String CONTEXT_SRC_DEVICE = 
            "net.floodlightcontroller.devicemanager.srcDevice"; 
    public static final String CONTEXT_DST_DEVICE = 
            "net.floodlightcontroller.devicemanager.dstDevice"; 
    public static final String CONTEXT_ORIG_DST_DEVICE =
            "net.floodlightcontroller.devicemanager.origDstDevice";
    public static final FloodlightContextStore<IDevice> fcStore = 
        new FloodlightContextStore<IDevice>();
    public IDevice getDevice(Long deviceKey);
    public IDevice findDevice(@Nonnull MacAddress macAddress, VlanVid vlan,
                              @Nonnull IPv4Address ipv4Address, @Nonnull IPv6Address ipv6Address,
                              @Nonnull DatapathId switchDPID, @Nonnull OFPort switchPort)
                              throws IllegalArgumentException;
    public IDevice findClassDevice(@Nonnull IEntityClass entityClass,
                                   @Nonnull MacAddress macAddress, VlanVid vlan,
                                   @Nonnull IPv4Address ipv4Address, @Nonnull IPv6Address ipv6Address)
                                   throws IllegalArgumentException;
    public Collection<? extends IDevice> getAllDevices();
    public void addIndex(boolean perClass,
                         EnumSet<DeviceField> keyFields);
    public Iterator<? extends IDevice> queryDevices(@Nonnull MacAddress macAddress,
                                                    VlanVid vlan,
                                                    @Nonnull IPv4Address ipv4Address, 
                                                    @Nonnull IPv6Address ipv6Address,
                                                    @Nonnull DatapathId switchDPID,
                                                    @Nonnull OFPort switchPort);
    public Iterator<? extends IDevice> queryClassDevices(IEntityClass entityClass,
                                                         MacAddress macAddress,
                                                         VlanVid vlan,
                                                         IPv4Address ipv4Address, 
                                                         IPv6Address ipv6Address,
                                                         DatapathId switchDPID,
                                                         OFPort switchPort);
    public void addListener(IDeviceListener listener);
    public void addSuppressAPs(DatapathId swId, OFPort port);
    public void removeSuppressAPs(DatapathId swId, OFPort port);
    public Set<SwitchPort> getSuppressAPs();
}
