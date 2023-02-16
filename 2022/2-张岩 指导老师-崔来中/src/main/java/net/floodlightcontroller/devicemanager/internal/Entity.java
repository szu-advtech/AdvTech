package net.floodlightcontroller.devicemanager.internal;
import java.util.Date;
import javax.annotation.Nonnull;
import net.floodlightcontroller.core.web.serializers.IPv4Serializer;
import net.floodlightcontroller.core.web.serializers.IPv6Serializer;
import net.floodlightcontroller.core.web.serializers.DPIDSerializer;
import net.floodlightcontroller.core.web.serializers.OFPortSerializer;
import net.floodlightcontroller.core.web.serializers.VlanVidSerializer;
import net.floodlightcontroller.core.web.serializers.MacSerializer;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
public class Entity implements Comparable<Entity> {
    protected static int ACTIVITY_TIMEOUT = 30000;
    protected MacAddress macAddress;
    protected IPv4Address ipv4Address;
    protected IPv6Address ipv6Address;
    protected VlanVid vlan;
    protected DatapathId switchDPID;
    protected OFPort switchPort;
    protected Date lastSeenTimestamp;
    protected Date activeSince;
    public Entity(@Nonnull MacAddress macAddress, VlanVid vlan, @Nonnull IPv4Address ipv4Address, 
    		@Nonnull IPv6Address ipv6Address, @Nonnull DatapathId switchDPID, @Nonnull OFPort switchPort, 
                  @Nonnull Date lastSeenTimestamp) {
    	if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (switchDPID == null) {
    		throw new IllegalArgumentException("Switch DPID cannot be null. Try DatapathId.NONE if intention is 'no DPID'");
    	}
    	if (switchPort == null) {
    		throw new IllegalArgumentException("Switch port cannot be null. Try OFPort.ZERO if intention is 'no port'");
    	}
    	if (lastSeenTimestamp == null) {
    		throw new IllegalArgumentException("Last seen time stamp cannot be null. Try Entity.NO_DATE if intention is 'no time'");
    	}
        this.macAddress = macAddress;
        this.ipv4Address = ipv4Address;
        this.ipv6Address = ipv6Address;
        this.vlan = vlan;
        this.switchDPID = switchDPID;
        this.switchPort = switchPort;
        this.lastSeenTimestamp = lastSeenTimestamp;
        this.activeSince = lastSeenTimestamp;
    }
    @JsonSerialize(using=MacSerializer.class)
    public MacAddress getMacAddress() {
        return macAddress;
    }
    @JsonSerialize(using=IPv4Serializer.class)
    public IPv4Address getIpv4Address() {
        return ipv4Address;
    }
    @JsonSerialize(using=IPv6Serializer.class)
    public IPv6Address getIpv6Address() {
        return ipv6Address;
    }
    @JsonSerialize(using=VlanVidSerializer.class)
    public VlanVid getVlan() {
        return vlan;
    }
    @JsonSerialize(using=DPIDSerializer.class)
    public DatapathId getSwitchDPID() {
        return switchDPID;
    }
    @JsonSerialize(using=OFPortSerializer.class)
    public OFPort getSwitchPort() {
        return switchPort;
    }
    @JsonIgnore
    public boolean hasSwitchPort() {
        return (switchDPID != null && !switchDPID.equals(DatapathId.NONE) && switchPort != null && !switchPort.equals(OFPort.ZERO));
    }
    public Date getLastSeenTimestamp() {
        return lastSeenTimestamp;
    }
    public void setLastSeenTimestamp(Date lastSeenTimestamp) {
        if (activeSince.equals(Entity.NO_DATE) ||
        		(activeSince.getTime() + ACTIVITY_TIMEOUT) < lastSeenTimestamp.getTime())
            this.activeSince = lastSeenTimestamp;
        this.lastSeenTimestamp = lastSeenTimestamp;
    }
    public Date getActiveSince() {
        return activeSince;
    }
    public void setActiveSince(Date activeSince) {
        this.activeSince = activeSince;
    }
    @Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
				+ ((ipv4Address == null) ? 0 : ipv4Address.hashCode());
				+ ((ipv6Address == null) ? 0 : ipv6Address.hashCode());
				+ ((macAddress == null) ? 0 : macAddress.hashCode());
				+ ((switchDPID == null) ? 0 : switchDPID.hashCode());
				+ ((switchPort == null) ? 0 : switchPort.hashCode());
		return result;
	}
    @Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Entity other = (Entity) obj;
		if (hashCode() != obj.hashCode())
			return false;
		if (ipv4Address == null) {
			if (other.ipv4Address != null)
				return false;
		} else if (!ipv4Address.equals(other.ipv4Address))
			return false;
		if (ipv6Address == null) {
			if (other.ipv6Address != null)
				return false;
		} else if (!ipv6Address.equals(other.ipv6Address))
			return false;
		if (macAddress == null) {
			if (other.macAddress != null)
				return false;
		} else if (!macAddress.equals(other.macAddress))
			return false;
		if (switchDPID == null) {
			if (other.switchDPID != null)
				return false;
		} else if (!switchDPID.equals(other.switchDPID))
			return false;
		if (switchPort == null) {
			if (other.switchPort != null)
				return false;
		} else if (!switchPort.equals(other.switchPort))
			return false;
		if (vlan == null) {
			if (other.vlan != null)
				return false;
		} else if (!vlan.equals(other.vlan))
			return false;
		return true;
	}
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Entity [macAddress=");
        if (macAddress != null) {
            builder.append(macAddress.toString());
        } else {
            builder.append("null");
        }
        builder.append(", ipv4Address=");
        if (ipv4Address != null) {
            builder.append(ipv4Address.toString());
        } else {
            builder.append("null");
        }
        builder.append(", ipv6Address=");
        if (ipv4Address != null) {
            builder.append(ipv6Address.toString());
        } else {
            builder.append("null");
        }
        builder.append(", vlan=");
        if (vlan != null) {
            builder.append(vlan.getVlan());
        } else {
            builder.append("null");
        }
        builder.append(", switchDPID=");
        if (switchDPID != null) {
            builder.append(switchDPID.toString());
        } else {
            builder.append("null");
        }
        builder.append(", switchPort=");
        if (switchPort != null) {
            builder.append(switchPort.getPortNumber());
        } else {
            builder.append("null");
        }
        builder.append(", lastSeenTimestamp=");
        if (lastSeenTimestamp != null) {
            builder.append(lastSeenTimestamp == null? "null" : lastSeenTimestamp.toString());
        } else {
            builder.append("null");
        }
        builder.append(", activeSince=");
        if (activeSince != null) {
            builder.append(activeSince == null? "null" : activeSince.toString());
        } else {
            builder.append("null");
        }
        builder.append("]");
        return builder.toString();
    }
    @Override
    public int compareTo(Entity o) {
        if (macAddress.getLong() < o.macAddress.getLong()) return -1;
        if (macAddress.getLong() > o.macAddress.getLong()) return 1;
        int r;
        if (switchDPID == null)
            r = o.switchDPID == null ? 0 : -1;
        else if (o.switchDPID == null)
            r = 1;
        else
            r = switchDPID.compareTo(o.switchDPID);
        if (r != 0) return r;
        if (switchPort == null)
            r = o.switchPort == null ? 0 : -1;
        else if (o.switchPort == null)
            r = 1;
        else
            r = switchPort.compareTo(o.switchPort);
        if (r != 0) return r;
        if (ipv4Address == null)
            r = o.ipv4Address == null ? 0 : -1;
        else if (o.ipv4Address == null)
            r = 1;
        else
            r = ipv4Address.compareTo(o.ipv4Address);
        if (r != 0) return r;
        if (ipv6Address == null)
            r = o.ipv6Address == null ? 0 : -1;
        else if (o.ipv6Address == null)
            r = 1;
        else
            r = ipv6Address.compareTo(o.ipv6Address);
        if (r != 0) return r;
        if (vlan == null)
            r = o.vlan == null ? 0 : -1;
        else if (o.vlan == null)
            r = 1;
        else
            r = vlan.compareTo(o.vlan);
        if (r != 0) return r;
        return 0;
    }
}