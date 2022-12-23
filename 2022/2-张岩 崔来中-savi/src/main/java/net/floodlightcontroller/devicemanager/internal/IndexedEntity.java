package net.floodlightcontroller.devicemanager.internal;
import java.util.EnumSet;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.OFPort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
public class IndexedEntity {
    protected EnumSet<DeviceField> keyFields;
    protected Entity entity;
    private int hashCode = 0;
    protected static Logger logger =
            LoggerFactory.getLogger(IndexedEntity.class);
    public IndexedEntity(EnumSet<DeviceField> keyFields, Entity entity) {
        super();
        this.keyFields = keyFields;
        this.entity = entity;
    }
    public boolean hasNonZeroOrNonNullKeys() {
        for (DeviceField f : keyFields) {
            switch (f) {
                    return true;
                case IPv4:
                    if (!entity.ipv4Address.equals(IPv4Address.NONE)) return true;
                    break;
                case IPv6:
                	if (!entity.ipv6Address.equals(IPv6Address.NONE)) return true;
                    break;
                case SWITCH:
                    if (!entity.switchDPID.equals(DatapathId.NONE)) return true;
                    break;
                case PORT:
                    if (!entity.switchPort.equals(OFPort.ZERO)) return true;
                    break;
                    if (entity.vlan != null) return true;
                    break;
            }
        }
        return false;
    }
    @Override
    public int hashCode() {
        if (hashCode != 0) {
        	return hashCode;
        }
        final int prime = 31;
        hashCode = 1;
        for (DeviceField f : keyFields) {
            switch (f) {
                case MAC:
                        + (int) (entity.macAddress.getLong() ^ 
                                (entity.macAddress.getLong() >>> 32));
                    break;
                case IPv4:
                        + ((entity.ipv4Address == null) 
                            ? 0 
                            : entity.ipv4Address.hashCode());
                    break;
                case IPv6:
                        + ((entity.ipv6Address == null) 
                            ? 0 
                            : entity.ipv6Address.hashCode());
                    break;
                case SWITCH:
                        + ((entity.switchDPID == null) 
                            ? 0 
                            : entity.switchDPID.hashCode());
                    break;
                case PORT:
                        + ((entity.switchPort == null) 
                            ? 0 
                            : entity.switchPort.hashCode());
                    break;
                case VLAN:
                        + ((entity.vlan == null) 
                            ? 0 
                            : entity.vlan.hashCode());
                    break;
            }
        }
        return hashCode;
    }
    @Override
    public boolean equals(Object obj) {
       if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        IndexedEntity other = (IndexedEntity) obj;
        if (!keyFields.equals(other.keyFields))
            return false;
        for (IDeviceService.DeviceField f : keyFields) {
            switch (f) {
                case MAC:
                    if (!entity.macAddress.equals(other.entity.macAddress))
                        return false;
                    break;
                case IPv4:
                    if (entity.ipv4Address == null) {
                        if (other.entity.ipv4Address != null) return false;
                    } else if (!entity.ipv4Address.equals(other.entity.ipv4Address)) return false;
                    break;
                case IPv6:
                    if (entity.ipv6Address == null) {
                        if (other.entity.ipv6Address != null) return false;
                    } else if (!entity.ipv6Address.equals(other.entity.ipv6Address)) return false;
                    break;
                case SWITCH:
                    if (entity.switchDPID == null) {
                        if (other.entity.switchDPID != null) return false;
                    } else if (!entity.switchDPID.equals(other.entity.switchDPID)) return false;
                    break;
                case PORT:
                    if (entity.switchPort == null) {
                        if (other.entity.switchPort != null) return false;
                    } else if (!entity.switchPort.equals(other.entity.switchPort)) return false;
                    break;
                case VLAN:
                    if (entity.vlan == null) {
                        if (other.entity.vlan != null) return false;
                    } else if (!entity.vlan.equals(other.entity.vlan)) return false;
                    break;
            }
        }  
        return true;
    }
}